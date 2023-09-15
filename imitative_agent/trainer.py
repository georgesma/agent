import torch
from tqdm import tqdm

from lib.early_stopping import EarlyStopping
from lib.training_record import TrainingRecord, EpochMetrics


class Trainer:
    def __init__(
        self,
        nn,
        optimizers,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        losses_fn,
        max_epochs,
        patience,
        synthesizer,
        sound_scalers,
        checkpoint_path,
        device="cuda",
    ):
        self.nn = nn.to(device)
        self.optimizers = optimizers
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.losses_fn = losses_fn
        self.max_epochs = max_epochs
        self.patience = patience
        self.synthesizer = synthesizer
        self.sound_scalers = sound_scalers
        self.checkpoint_path = checkpoint_path
        self.device = device

    def train(self):
        training_record = TrainingRecord()
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )

        for epoch in range(1, self.max_epochs + 1):
            print("== Epoch %s ==" % epoch)

            train_metrics = self.epoch_train(self.train_dataloader)
            training_record.save_epoch_metrics("train", train_metrics)

            validation_metrics = self.epoch_evaluate(self.validation_dataloader)
            training_record.save_epoch_metrics("validation", validation_metrics)

            if self.test_dataloader is not None:
                test_metrics = self.epoch_evaluate(self.test_dataloader)
                training_record.save_epoch_metrics("test", test_metrics)

            early_stopping(
                validation_metrics.metrics["inverse_model_repetition_error"], self.nn
            )

            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                print()

        self.nn.load_state_dict(torch.load(self.checkpoint_path))
        return training_record.record

    def epoch_train(self, dataloader):
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        for batch in tqdm(dataloader, total=nb_batch, leave=False):
            sound_seqs, seqs_len, seqs_mask = batch
            sound_seqs = sound_seqs.to("cuda")
            seqs_mask = seqs_mask.to("cuda")

            self.step_direct_model(
                sound_seqs, seqs_len, seqs_mask, epoch_record, is_training=True
            )
            self.step_inverse_model(
                sound_seqs, seqs_len, seqs_mask, epoch_record, is_training=True
            )

        return epoch_record

    def epoch_evaluate(self, dataloader):
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        self.nn.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, total=nb_batch, leave=False):
                sound_seqs, seqs_len, seqs_mask = batch
                sound_seqs = sound_seqs.to("cuda")
                seqs_mask = seqs_mask.to("cuda")

                self.step_direct_model(
                    sound_seqs, seqs_len, seqs_mask, epoch_record, is_training=False
                )
                self.step_inverse_model(
                    sound_seqs, seqs_len, seqs_mask, epoch_record, is_training=False
                )

        return epoch_record

    def step_direct_model(
        self, sound_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        if is_training:
            self.nn.inverse_model.eval()
            self.nn.direct_model.train()
            self.nn.direct_model.requires_grad_(True)

        with torch.no_grad():
            art_seqs_estimated = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(art_seqs_estimated)
        sound_seqs_produced = self.sound_scalers["synthesizer"].inverse_transform(
            sound_seqs_produced
        )
        sound_seqs_produced = self.sound_scalers["agent"].transform(sound_seqs_produced)

        if is_training:
            self.optimizers["direct_model"].zero_grad()
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)
        direct_model_loss = self.losses_fn["mse"](
            sound_seqs_estimated, sound_seqs_produced, seqs_mask
        )
        if is_training:
            direct_model_loss.backward()
            self.optimizers["direct_model"].step()

        epoch_record.add("direct_model_estimation_error", direct_model_loss.item())

    def step_inverse_model(
        self, sound_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        # Inverse model training/evaluation
        # (inverse model estimation → direct model estimation vs. perceived sound)
        if is_training:
            self.nn.inverse_model.train()
            self.nn.direct_model.eval()
            self.nn.direct_model.requires_grad_(False)

            self.optimizers["inverse_model"].zero_grad()

        art_seqs_estimated = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)

        inverse_total, inverse_estimation_error, inverse_jerk = self.losses_fn[
            "inverse_model"
        ](art_seqs_estimated, sound_seqs_estimated, sound_seqs, seqs_mask)
        if is_training:
            inverse_total.backward()
            self.optimizers["inverse_model"].step()

        epoch_record.add(
            "inverse_model_estimation_error", inverse_estimation_error.item()
        )
        epoch_record.add("inverse_model_jerk", inverse_jerk.item())

        # Inverse model repetition error
        # (inverse model estimation → synthesizer vs. perceived sound)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(
            art_seqs_estimated.detach()
        )
        repetition_error = self.losses_fn["mse"](
            sound_seqs_produced, sound_seqs, seqs_mask
        )
        epoch_record.add("inverse_model_repetition_error", repetition_error.item())
