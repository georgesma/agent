import torch
from tqdm import tqdm

from lib.early_stopping import EarlyStopping
from lib.training_record import TrainingRecord, EpochMetrics
from lib.nn.pad_seqs_frames import pad_seqs_frames


class Trainer:
    def __init__(
        self,
        nn,
        optimizer,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        loss_fn,
        max_epochs,
        patience,
        checkpoint_path,
        device= "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.nn = nn.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.patience = patience
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
                validation_metrics.metrics["total_loss"], self.nn
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

        self.nn.train()
        for batch in tqdm(dataloader, total=nb_batch, leave=False):
            data_seqs, speaker_seqs, seqs_len, seqs_mask = batch
            data_seqs = data_seqs.to(self.device)
            speaker_seqs = speaker_seqs.to(self.device)
            seqs_mask = seqs_mask.to(self.device)

            self.step_quantizer(
                data_seqs, speaker_seqs, seqs_mask, epoch_record, is_training=True
            )

        return epoch_record

    def epoch_evaluate(self, dataloader):
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        self.nn.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, total=nb_batch, leave=False):
                data_seqs, speaker_seqs, seqs_len, seqs_mask = batch
                data_seqs = data_seqs.to(self.device)
                speaker_seqs = speaker_seqs.to(self.device)
                seqs_mask = seqs_mask.to(self.device)

                self.step_quantizer(
                    data_seqs, speaker_seqs, seqs_mask, epoch_record, is_training=False
                )

        return epoch_record

    def step_quantizer(
        self, data_seqs, speaker_seqs, seqs_mask, epoch_record, is_training
    ):
        padded_data_seqs = pad_seqs_frames(data_seqs, self.nn.frame_padding)

        if is_training:
            self.optimizer.zero_grad()
        padded_data_seqs_pred, vq_loss_seqs, quantized_latent_seqs, _, _ = self.nn(
            padded_data_seqs, speaker_seqs, pad_io=False
        )
        total_loss, reconstruction_error, vq_loss = self.loss_fn(
            padded_data_seqs_pred, padded_data_seqs, vq_loss_seqs, seqs_mask
        )
        if is_training:
            total_loss.backward()
            self.optimizer.step()

        epoch_record.add("total_loss", total_loss.item())
        epoch_record.add("reconstruction_error", reconstruction_error.item())
        epoch_record.add("vq_loss", vq_loss.item())
