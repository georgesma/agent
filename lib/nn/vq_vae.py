import torch
from torch import nn

from lib.nn.feedforward import FeedForward
from lib.nn.vector_quantizer import VectorQuantizer
from lib.nn.pad_seqs_frames import pad_seqs_frames, unpad_seqs_frames


class VQVAE(nn.Module):
    def __init__(
        self,
        x_dim,
        frame_padding,
        hidden_dims,
        activation,
        embedding_dim,
        num_embeddings,
        commitment_cost,
        nb_speakers,
        dropout_p,
        batch_norm,
    ):
        super(VQVAE, self).__init__()

        self.frame_padding = frame_padding
        self.padded_x_dim = x_dim * (1 + 2 * frame_padding)

        self.encoder = FeedForward(
            x_dim=self.padded_x_dim,
            y_dim=embedding_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_p=dropout_p,
            batch_norm=batch_norm,
        )

        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = FeedForward(
            x_dim=embedding_dim + nb_speakers,
            y_dim=self.padded_x_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_p=dropout_p,
            batch_norm=batch_norm,
        )

    def forward(self, seqs, speaker_id, pad_io=True):
        vq_loss, quantized_latent, quantized_index, encoder_output = self.encode(
            seqs, pad_input=pad_io
        )
        seqs_pred = self.decode(quantized_latent, speaker_id, unpad_output=pad_io)
        return seqs_pred, vq_loss, quantized_latent, quantized_index, encoder_output

    def encode(self, seqs, pad_input=True):
        if pad_input:
            seqs = pad_seqs_frames(seqs, self.frame_padding)
        encoder_output = self.encoder(seqs)
        vq_loss, quantized_latent, perplexity, quantized_index = self.vector_quantizer(
            encoder_output
        )
        return vq_loss, quantized_latent, quantized_index, encoder_output

    def decode(self, decoder_input, speaker_id, unpad_output=True):
        decoder_input = torch.cat((decoder_input, speaker_id), dim=-1)
        seqs_pred = self.decoder(decoder_input)
        if unpad_output:
            seqs_pred = unpad_seqs_frames(seqs_pred, self.frame_padding)
        return seqs_pred
