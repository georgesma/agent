import torch
from torch import nn


class DataScaler(nn.Module):
    def __init__(self, dim):
        super(DataScaler, self).__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(dim, dtype=torch.float32))

    def set_params(self, std, mean):
        if std.device != self.std.device:
            std = std.to(self.std.device)
            mean = std.to(self.std.device)
        self.std = std
        self.mean = mean

    def fit(self, data):
        std, mean = torch.std_mean(data, dim=0)
        self.set_params(std, mean)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    @staticmethod
    def from_standard_scaler(standard_scaler):
        std = torch.FloatTensor(standard_scaler.var_)
        mean = torch.FloatTensor(standard_scaler.mean_)
        data_scaler = DataScaler(standard_scaler.n_features_in_)
        data_scaler.set_params(std, mean)
        return data_scaler
