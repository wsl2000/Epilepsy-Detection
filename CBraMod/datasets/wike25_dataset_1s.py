import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import pickle

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.mode = mode
        batch_dir = os.path.join(data_dir, mode)
        self.batch_files = [os.path.join(batch_dir, file) for file in os.listdir(batch_dir) if file.endswith('.pkl')]
        
        # 预加载所有批次数据到内存
        self.all_data = []
        self.all_labels = []
        
        for batch_file in self.batch_files:
            with open(batch_file, 'rb') as f:
                batch_dict = pickle.load(f)
                batch_data = batch_dict['X']  # [batch_size, 19, 1, 200]
                batch_labels = batch_dict['y']  # [batch_size]
                
                for i in range(len(batch_data)):
                    self.all_data.append(batch_data[i])  # [19, 1, 200]
                    self.all_labels.append(batch_labels[i])

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data = self.all_data[idx]  # [19, 1, 200]
        label = self.all_labels[idx]
        return data, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])  # [batch_size, 19, 1, 200]
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
