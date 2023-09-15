import torch
from tqdm import tqdm

from lib.early_stopping import EarlyStopping
from lib.training_record import TrainingRecord, EpochMetrics


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
        device="cuda",
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

            early_stopping(validation_metrics.metrics["total"], self.nn)

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
            art_seqs, sound_seqs, seqs_len, seqs_mask = batch
            art_seqs = art_seqs.to("cuda")
            sound_seqs = sound_seqs.to("cuda")
            seqs_mask = seqs_mask.to("cuda")

            self.optimizer.zero_grad()
            sound_seqs_pred = self.nn(art_seqs)
            reconstruction_loss = self.loss_fn(sound_seqs_pred, sound_seqs, seqs_mask)
            reconstruction_loss.backward()
            self.optimizer.step()

            epoch_record.add("total", reconstruction_loss.item())

        return epoch_record

    def epoch_evaluate(self, dataloader):
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        self.nn.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, total=nb_batch, leave=False):
                art_seqs, sound_seqs, seqs_len, seqs_mask = batch
                art_seqs = art_seqs.to("cuda")
                sound_seqs = sound_seqs.to("cuda")
                seqs_mask = seqs_mask.to("cuda")

                sound_seqs_pred = self.nn(art_seqs)
                reconstruction_loss = self.loss_fn(
                    sound_seqs_pred, sound_seqs, seqs_mask
                )

                epoch_record.add("total", reconstruction_loss.item())

        return epoch_record
