import torch

class DataModule:
    
    def __init__(self, root = '../data', num_workers = 4):
        self.root = root
        self.num_workers = num_workers

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train = True)

    def val_dataloader(self):
        return self.get_dataloader(train = False)

    def tensor_dataloader(self, tensors, indices, train):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle = train)