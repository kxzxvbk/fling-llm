from torch.utils.data import Dataset


class CyclingDataset(Dataset):
    def __init__(self, data: Dataset):
        self.data = data
        self.start_idx = 0
        self.idxes = []

    def update(self, train_num: int):
        self.idxes = []
        for i in range(train_num + 1):
            self.idxes.append(i + self.start_idx % len(self.data))

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, idx):
        return self.data[self.idxes[idx]]
