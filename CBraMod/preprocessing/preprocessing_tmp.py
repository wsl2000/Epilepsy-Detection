import os
import pickle
import numpy as np
import csv
import random
import scipy.io as sio
import matplotlib.pyplot as plt

# 目标通道顺序
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4',
    'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]

input_folder = r"D:\datasets\eeg\dataset_dir_original\shared_data\training"
output_folder = r"D:\datasets\eeg\dataset_processed\shared_data"

for sub in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

def load_one_reference(folder: str):
    """
    只读取REFERENCE.csv第一条，并加载对应.mat文件，返回数据
    """
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rec_id = row[0]
            label = (int(row[1]), float(row[2]), float(row[3]))
            mat_file = os.path.join(folder, rec_id + '.mat')
            if not os.path.exists(mat_file):
                print(f"{mat_file} 不存在，跳过。")
                continue
            eeg_data = sio.loadmat(mat_file, simplify_cells=True)
            ch_names = [x.strip(' ') for x in eeg_data.get('channels')]
            data = eeg_data.get('data')
            fs = eeg_data.get('fs')
            return rec_id, ch_names, data, fs, label
    raise RuntimeError("没有可用数据！")

def normalize_per_channel(segment):
    """
    对每个通道进行归一化（z-score: 均值0 方差1）
    输入: segment [n_channel, n_samples]
    输出: 同shape归一化后的数据
    """
    mean = segment.mean(axis=1, keepdims=True)
    std = segment.std(axis=1, keepdims=True)
    std[std==0] = 1.0
    return (segment - mean) / std

def process_and_show_one(
    rec_id,
    ch_names,
    data,
    fs,
    label,
    subfolder="train"
):
    """
    处理一条数据，分割第一个片段并保存为pkl，同时可视化
    """
    out_dir = os.path.join(output_folder, subfolder)
    seizure_present, onset, offset = label
    onset_sample = int(onset * fs)
    offset_sample = int(offset * fs)
    total_len = data.shape[1]

    # 按照目标通道顺序组织通道
    channel_data = []
    for ch in TARGET_CHANNELS:
        if ch in ch_names:
            idx = ch_names.index(ch)
            channel_data.append(data[idx])
        else:
            channel_data.append(np.zeros(total_len, dtype=np.float32))
    channel_data = np.stack(channel_data, axis=0)

    segment_len = int(10 * fs)
    # 找第一个完整的片段
    for i in range(0, total_len, segment_len):
        segment = channel_data[:, i:i+segment_len]
        if segment.shape[1] == segment_len:
            # 归一化
            segment_norm = normalize_per_channel(segment)
            if seizure_present and not (i+segment_len < onset_sample or i > offset_sample):
                y = 1
            else:
                y = 0
            save_name = f"{rec_id}-{i}.pkl"
            save_path = os.path.join(out_dir, save_name)
            with open(save_path, 'wb') as f:
                pickle.dump({"X": segment_norm, "y": y}, f)
            print(f"已保存: {save_path} 标签:{y} 已归一化")

            # 可视化
            plt.figure(figsize=(12, 7))
            offset_step = 6  # 归一化后缩小偏移量以便显示
            for ch in range(segment_norm.shape[0]):
                plt.plot(segment_norm[ch] + ch * offset_step, label=TARGET_CHANNELS[ch])
            plt.title(f"{rec_id}-{i}.pkl (label={y}) [已归一化]")
            plt.xlabel('采样点')
            plt.ylabel('归一化信号+通道偏移')
            plt.legend(loc='upper right', bbox_to_anchor=(1.18, 1.0))
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    random.seed(42)
    rec_id, ch_names, data, fs, label = load_one_reference(input_folder)
    process_and_show_one(rec_id, ch_names, data, fs, label, subfolder="train")