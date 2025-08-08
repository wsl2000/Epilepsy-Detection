import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from scipy import signal

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            target_channels=19,  # 目标通道数
            target_length=2000,  # 目标长度
    ):
        super(CustomDataset, self).__init__()
        self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]
        self.target_channels = target_channels
        self.target_length = target_length
        
        # 分析数据集中的通道数分布
        self._analyze_channel_distribution()

    def _analyze_channel_distribution(self):
        """分析数据集中的通道数分布"""
        channel_counts = {}
        sample_files = self.files[:min(100, len(self.files))]  # 采样分析
        
        for file in sample_files:
            try:
                data_dict = pickle.load(open(file, 'rb'))
                num_channels = data_dict.get('num_channels', data_dict['X'].shape[0])
                channel_counts[num_channels] = channel_counts.get(num_channels, 0) + 1
            except Exception as e:
                print(f"分析文件 {file} 时出错: {e}")
        
        print("数据集通道数分布:")
        for channels, count in sorted(channel_counts.items()):
            print(f"  {channels} 通道: {count} 个文件")

    def _pad_or_crop_channels(self, data, current_channels):
        """调整通道数到目标通道数"""
        if current_channels == self.target_channels:
            return data
        elif current_channels > self.target_channels:
            # 如果通道数过多，选择前target_channels个通道
            return data[:self.target_channels, :]
        else:
            # 如果通道数不足，用零填充
            padded_data = np.zeros((self.target_channels, data.shape[1]), dtype=data.dtype)
            padded_data[:current_channels, :] = data
            return padded_data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        try:
            data_dict = pickle.load(open(file, 'rb'))
            data = data_dict['X']
            label = data_dict['y']
            
            # 获取当前通道数
            current_channels = data.shape[0]
            
            # 重采样到目标长度
            data = signal.resample(data, self.target_length, axis=1)
            
            # 调整通道数
            data = self._pad_or_crop_channels(data, current_channels)
            
            # 重塑为 (channels, time_segments, samples_per_segment)
            samples_per_segment = self.target_length // 10  # 每段的采样点数
            data = data.reshape(self.target_channels, 10, samples_per_segment)
            
            # 单位转换
            data = data * 1e6   # 如果原文件单位是 Volt；若本来就是 µV, 注释掉这行

            # 通道级标准化
            mu = np.mean(data, axis=(1, 2), keepdims=True)  # shape (channels, 1, 1)
            std = np.std(data, axis=(1, 2), keepdims=True) + 1e-8
            data = (data - mu) / std
            
            return data, label
            
        except Exception as e:
            print(f"加载文件 {file} 时出错: {e}")
            # 返回默认数据避免训练中断
            default_data = np.zeros((self.target_channels, 10, self.target_length // 10), dtype=np.float32)
            return default_data, 0

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        # 可以通过参数配置目标通道数
        target_channels = 19
        target_length = 2000
        
        train_set = CustomDataset(self.datasets_dir, mode='train', 
                                target_channels=target_channels, target_length=target_length)
        val_set = CustomDataset(self.datasets_dir, mode='val',
                              target_channels=target_channels, target_length=target_length)
        test_set = CustomDataset(self.datasets_dir, mode='test',
                               target_channels=target_channels, target_length=target_length)
        
        print(f"数据集大小: 训练={len(train_set)}, 验证={len(val_set)}, 测试={len(test_set)}")
        print(f"总计: {len(train_set) + len(val_set) + len(test_set)}")
        
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=3,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=3,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=3,
            ),
        }
        return data_loader
