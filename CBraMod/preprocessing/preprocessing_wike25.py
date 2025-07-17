import os
import pickle
import numpy as np
import csv
import random
import scipy.io as sio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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
output_folder = r"D:\datasets\eeg\dataset_processed\shared_data2"

for sub in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

def load_all_references(folder: str):
    refs = []
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rec_id = row[0]
            label = (int(row[1]), float(row[2]), float(row[3]))
            refs.append((rec_id, label))
    return refs

def normalize_per_channel(segment):
    mean = segment.mean(axis=1, keepdims=True)
    std = segment.std(axis=1, keepdims=True)
    std[std==0] = 1.0
    return (segment - mean) / std

def scale_per_channel(segment_norm, amp=100.0):
    # 每通道最大绝对值缩放到amp
    seg = segment_norm.copy()
    for i in range(seg.shape[0]):
        max_v = np.max(np.abs(seg[i]))
        if max_v > 1e-6:
            seg[i] = seg[i] * (amp / max_v)
    return seg

def visualize_segment_ax(segment_norm, save_path, y):
    """
    可视化一个归一化后的片段，并保存图片
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    offset_step = 200
    for ch in range(segment_norm.shape[0]):
        ax.plot(segment_norm[ch] + ch * offset_step, label=TARGET_CHANNELS[ch])
    ax.set_title(f'{os.path.basename(save_path)} (label={y})')
    ax.set_xlabel('采样点')
    ax.set_ylabel('信号+通道偏移')
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.0))
    plt.tight_layout()
    img_path = save_path.replace('.pkl', '.png')
    plt.savefig(img_path)
    plt.close(fig)

# 可视化开关（True=生成图片，False=只保存pkl不保存图片）
VISUALIZE = False

def process_one_record(rec_id, label, ch_names, data, fs, subfolder):
    out_dir = os.path.join(output_folder, subfolder)
    seizure_present, onset, offset = label
    onset_sample = int(onset * fs)
    offset_sample = int(offset * fs)
    total_len = data.shape[1]

    channel_data = []
    for ch in TARGET_CHANNELS:
        if ch in ch_names:
            idx = ch_names.index(ch)
            channel_data.append(data[idx])
        else:
            channel_data.append(np.zeros(total_len, dtype=np.float32))
    channel_data = np.stack(channel_data, axis=0)

    segment_len = int(10 * fs)
    for i in range(0, total_len, segment_len):
        segment = channel_data[:, i:i+segment_len]
        if segment.shape[1] == segment_len:
            segment_norm = normalize_per_channel(segment)
            segment_scaled = scale_per_channel(segment_norm, amp=100.0)
            if seizure_present and not (i+segment_len < onset_sample or i > offset_sample):
                y = 1
            else:
                y = 0
            save_name = f"{rec_id}-{i}.pkl"
            save_path = os.path.join(out_dir, save_name)
            with open(save_path, 'wb') as f:
                pickle.dump({"X": segment_scaled, "y": y}, f)
            if VISUALIZE:
                visualize_segment_ax(segment_scaled, save_path, y)
    # 正样本增强（5秒滑窗）
    if seizure_present:
        for idx, start in enumerate(range(max(0, onset_sample-int(fs)), min(offset_sample+int(fs), total_len-segment_len+1), int(5*fs))):
            segment = channel_data[:, start: start+segment_len]
            if segment.shape[1] == segment_len:
                segment_norm = normalize_per_channel(segment)
                segment_scaled = scale_per_channel(segment_norm, amp=100.0)
                save_name = f"{rec_id}-s-add-{idx}-{start}.pkl"
                save_path = os.path.join(out_dir, save_name)
                with open(save_path, 'wb') as f:
                    pickle.dump({"X": segment_scaled, "y": 1}, f)
                if VISUALIZE:
                    visualize_segment_ax(segment_scaled, save_path, 1)

def process_record_wrapper(args):
    rec_id, label, subfolder = args
    mat_file = os.path.join(input_folder, rec_id + '.mat')
    if not os.path.exists(mat_file):
        print(f"{mat_file} 不存在，跳过。")
        return
    eeg_data = sio.loadmat(mat_file, simplify_cells=True)
    ch_names = [x.strip(' ') for x in eeg_data.get('channels')]
    data = eeg_data.get('data')
    fs = eeg_data.get('fs')
    process_one_record(rec_id, label, ch_names, data, fs, subfolder)



PROCESS_RATIO = 0.5  # 这里设置为40%，你可以根据需要修改

if __name__ == "__main__":
    random.seed(42)
    refs = load_all_references(input_folder)
    random.shuffle(refs)
    N = len(refs)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    train_refs = refs[:n_train]
    val_refs = refs[n_train:n_train+n_val]
    test_refs = refs[n_train+n_val:]

    # 计算要处理的数量
    def get_partial(refs):
        n = int(len(refs) * PROCESS_RATIO)
        return refs[:n]

    train_refs = get_partial(train_refs)
    val_refs = get_partial(val_refs)
    test_refs = get_partial(test_refs)

    all_args = []
    for rec_id, label in train_refs:
        all_args.append((rec_id, label, "train"))
    for rec_id, label in val_refs:
        all_args.append((rec_id, label, "val"))
    for rec_id, label in test_refs:
        all_args.append((rec_id, label, "test"))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        tasks = [executor.submit(process_record_wrapper, args) for args in all_args]
        for _ in tqdm(as_completed(tasks), total=len(tasks)):
            pass

    print("全部分割、保存幅度放大pkl完成。")