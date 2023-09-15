from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SimpleLSTM(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, num_layers, dropout_p, bidirectional):
        super(SimpleLSTM, self).__init__()
        self.build(x_dim, y_dim, hidden_size, num_layers, dropout_p, bidirectional)

    def build(self, x_dim, y_dim, hidden_size, num_layers, dropout_p, bidirectional):
        self.lstm = nn.LSTM(
            x_dim,
            hidden_size,
            num_layers,
            dropout=dropout_p,
            bidirectional=bidirectional,
            batch_first=True,
        )
        d = 2 if bidirectional else 1
        self.output_layer = nn.Linear(d * hidden_size, y_dim)

    def forward(self, seqs, seqs_len=None):
        if seqs_len != None:
            seqs = pack_padded_sequence(seqs, seqs_len, batch_first=True)
        lstm_output, _ = self.lstm(seqs)
        if seqs_len != None:
            lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        seqs_pred = self.output_layer(lstm_output)
        return seqs_pred
