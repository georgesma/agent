from contextlib import nullcontext
from tqdm import tqdm
import torch

from lib.early_stopping import EarlyStopping
from lib.training_record import TrainingRecord, EpochMetrics
from lib.nn.pad_seqs_frames import pad_seqs_frames


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
        self.train_model_part(
            training_record, self.epoch_inverse_model, "inverse_model_repetition_error"
        )
        return training_record.record

    def train_model_part(self, training_record, epoch_fn, early_stopping_metric):
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )

        for epoch in range(1, self.max_epochs + 1):
            print("== Epoch %s ==" % epoch)

            train_metrics = epoch_fn(self.train_dataloader, is_training=True)
            training_record.save_epoch_metrics("train", train_metrics)

            validation_metrics = epoch_fn(self.validation_dataloader, is_training=False)
            training_record.save_epoch_metrics("validation", validation_metrics)

            if self.test_dataloader is not None:
                test_metrics = epoch_fn(self.test_dataloader, is_training=False)
                training_record.save_epoch_metrics("test", test_metrics)

            early_stopping(validation_metrics.metrics[early_stopping_metric], self.nn)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                print()

        self.nn.load_state_dict(torch.load(self.checkpoint_path))

    def epoch_inverse_model(self, dataloader, is_training):
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        if not is_training:
            self.nn.eval()

        with torch.no_grad() if not is_training else nullcontext():
            for batch in tqdm(dataloader, total=nb_batch, leave=False):
                sound_seqs, speaker_seqs, seqs_len, seqs_mask = batch
                sound_seqs = sound_seqs.to("cuda")
                speaker_seqs = speaker_seqs.to("cuda")
                seqs_mask = seqs_mask.to("cuda")

                with torch.no_grad():
                    _, _, sound_unit_seqs, _, _ = self.nn.sound_quantizer(
                        sound_seqs, speaker_seqs
                    )
                    sound_unit_seqs = sound_unit_seqs.detach()
                if "direct_model" in self.optimizers:
                    self.step_direct_model(
                        sound_unit_seqs,
                        seqs_len,
                        seqs_mask,
                        epoch_record,
                        is_training=is_training,
                    )
                self.step_inverse_model(
                    sound_unit_seqs,
                    seqs_len,
                    seqs_mask,
                    epoch_record,
                    is_training=is_training,
                )

        return epoch_record

    def step_direct_model(
        self, sound_unit_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        if is_training:
            self.nn.inverse_model.eval()
            self.nn.direct_model.train()
            self.nn.direct_model.requires_grad_(True)

        with torch.no_grad():
            art_seqs_estimated = self.nn.inverse_model(
                sound_unit_seqs, seqs_len=seqs_len
            )
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
        self, sound_unit_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        # Inverse model training/evaluation
        # (inverse model estimation → direct model estimation vs. perceived sound)
        if is_training:
            self.nn.inverse_model.train()
            self.nn.direct_model.eval()
            self.nn.direct_model.requires_grad_(False)
            self.nn.sound_quantizer.eval()
            self.nn.sound_quantizer.requires_grad_(False)

            self.optimizers["inverse_model"].zero_grad()

        art_seqs_estimated = self.nn.inverse_model(sound_unit_seqs, seqs_len=seqs_len)
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)
        _, _, _, sound_unit_seqs_estimated = self.nn.sound_quantizer.encode(
            sound_seqs_estimated
        )

        inverse_total, inverse_estimation_error, inverse_jerk = self.losses_fn[
            "inverse_model"
        ](art_seqs_estimated, sound_unit_seqs_estimated, sound_unit_seqs, seqs_mask)
        if is_training:
            inverse_total.backward()
            self.optimizers["inverse_model"].step()

        epoch_record.add(
            "inverse_model_estimation_error", inverse_estimation_error.item()
        )
        epoch_record.add("inverse_model_jerk", inverse_jerk.item())

        # Inverse model repetition error
        # (inverse model estimation → synthesizer → sound quantizer encoder
        # vs. perceived sound → sound quantizer encoder → sound quantizer quantization)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(
            art_seqs_estimated.detach()
        )
        with torch.no_grad():
            _, sound_unit_seqs_produced, _, _ = self.nn.sound_quantizer.encode(
                sound_seqs_produced
            )
        repetition_error = self.losses_fn["mse"](
            sound_unit_seqs_produced, sound_unit_seqs, seqs_mask
        )
        epoch_record.add("inverse_model_repetition_error", repetition_error.item())
