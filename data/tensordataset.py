import numpy as np
import math

class TensorDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class DataLoader:
    def __init__(self, dataset, batch_dims, shuffle=False, drop_last=False):
        self.dataset = dataset
        assert isinstance(batch_dims, int)
        self.batch_dims = batch_dims
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        bs = self.batch_dims
        N = math.floor(len(self.dataset) / bs)
        return N if self.drop_last else math.ceil(len(self.dataset) / bs) #N + 1

    def __iter__(self):
        return DatasetIterator(self)

    def __next__(self):
        indices = np.random.choice(len(self.dataset), size=(self.batch_dims,))
        indices = indices.tolist()
        return self.dataset[indices]


class DatasetIterator:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        if self.dataloader.shuffle:
            self.indices = np.random.permutation(len(self.dataloader.dataset))
        else:
            self.indices = np.arange(len(self.dataloader.dataset))
        self.bs = self.dataloader.batch_dims
        self.N = math.floor(len(dataloader.dataset) / self.bs)
        self.n = 0

    def __next__(self):
        if self.n < self.N:
            indices = self.indices[self.bs * self.n : self.bs * (self.n + 1)].tolist()
            batch = self.dataloader.dataset[indices]
            self.n = self.n + 1
        elif (self.n == self.N) and not self.dataloader.drop_last:
            indices = self.indices[self.bs * self.n :].tolist()
            batch = self.dataloader.dataset[indices]
            self.n = self.n + 1
            # TODO: This only works for 1D batch dims rn
        else:
            raise StopIteration

        return batch


# TODO: assumes 1d batch_dims
class SubDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return self.indices.shape[0]


def random_split(dataset, lengths):
    if lengths is None:
        return dataset, dataset, dataset
    elif sum(lengths) == len(dataset):
        pass
    elif sum(lengths) == 1:
        lengths = [int(l * len(dataset)) for l in lengths]
        lengths[-1] = len(dataset) - int(sum(lengths[:-1]))
    else:
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset"
        )

    indicies = np.random.permutation(len(dataset))
    return [
        SubDataset(dataset, indicies[sum(lengths[:i]) : sum(lengths[: i + 1])])
        for i in range(len(lengths))
    ]