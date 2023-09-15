from torch import nn


class CommunicativeAgentNN(nn.Module):
    def __init__(self, inverse_model, direct_model, sound_quantizer, art_quantizer):
        super(CommunicativeAgentNN, self).__init__()
        self.inverse_model = inverse_model
        self.direct_model = direct_model
        self.sound_quantizer = sound_quantizer
        self.art_quantizer = art_quantizer

    # def forward(self, sound_seqs):
    #     art_seqs_pred = self.inverse_model(sound_seqs)
    #     sound_seqs_pred = self.direct_model(art_seqs_pred)
    #     return sound_seqs_pred, art_seqs_pred
