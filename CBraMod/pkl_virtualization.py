import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 指定pkl文件目录
pkl_dir = r'D:\\datasets\\eeg\\dataset_processed\\CHB-MIT_seg\val'  # 改成你的路径
TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4',
    'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]
pkl_dir = r'D:\\datasets\\eeg\\dataset_processed\\shared_data\val'  # 改成你的路径

pkl_files = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')))
if not pkl_files:
    print("没有找到pkl文件！")
    exit(0)

fig, ax = plt.subplots(figsize=(12, 10))

def find_next_label1(start_idx, direction=1):
    n = len(pkl_files)
    i = start_idx
    for _ in range(n):
        with open(pkl_files[i], 'rb') as f:
            data = pickle.load(f)
        if data['y'] == 1:
            return i
        i = (i + direction) % n
    return None  # 没有label=1的

def plot_idx(i):
    ax.clear()
    with open(pkl_files[i], 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    for ch in range(X.shape[0]):
        ax.plot(X[ch] + ch * 200, label=TARGET_CHANNELS[ch])
    ax.set_title(f'{os.path.basename(pkl_files[i])} (label={y})')
    ax.set_xlabel('采样点')
    ax.set_ylabel('信号+通道偏移')
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.0))
    plt.tight_layout()
    fig.canvas.draw()

# 初始化idx为第一个label=1的
idx = find_next_label1(0, 1)
if idx is None:
    print("没有找到label=1的pkl文件！")
    exit(0)

def on_key(event):
    global idx
    if event.key == 'right':
        next_idx = find_next_label1((idx + 1) % len(pkl_files), 1)
        if next_idx is not None:
            idx = next_idx
            plot_idx(idx)
    elif event.key == 'left':
        prev_idx = find_next_label1((idx - 1) % len(pkl_files), -1)
        if prev_idx is not None:
            idx = prev_idx
            plot_idx(idx)

plot_idx(idx)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()