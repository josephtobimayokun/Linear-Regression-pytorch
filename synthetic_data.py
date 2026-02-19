import torch
from module.data_module import DataModule

class SyntheticRegressionData(DataModule):
    
    def __init__(self, w, b, noise = 0.01, num_train = 1000, num_val = 1000, batch_size = 32):
        self.w = w
        self.b = b
        self.noise = noise
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = self.X @ w.reshape((-1, 1)) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, self.num_train + self.num_val)
        return self.tensor_dataloader((self.X, self.y), i, train)