import os
import pickle
import numpy as np
import csv
import random
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.signal

# 目标通道顺序
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4',
    'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]

input_folder = r"D:\datasets\eeg\dataset_dir_original\shared_data\training"
output_folder = r"D:\datasets\eeg\dataset_processed\1s_256_batchsize"
PROCESS_RATIO = 1  # 这里设置为100%，你可以根据需要修改
batch_size = 256

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

def process_one_record(rec_id, label, ch_names, data, fs):
    """处理单个记录，返回生成器以节省内存"""
    seizure_present, onset, offset = label
    total_len = data.shape[1]

    channel_data = []
    for ch in TARGET_CHANNELS:
        if ch in ch_names:
            idx = ch_names.index(ch)
            channel_data.append(data[idx])
        else:
            channel_data.append(np.zeros(total_len, dtype=np.float32))
    channel_data = np.stack(channel_data, axis=0)
    
    # 将数据乘以1000，从[-0.01, 0.01] μV转换为[-10, 10] μV
    channel_data = channel_data * 1000

    # 重采样到200Hz
    target_fs = 200
    target_len = int(total_len * target_fs / fs)
    channel_data_resampled = scipy.signal.resample(channel_data, target_len, axis=1)

    # onset/offset重新对齐
    onset_sample = int(onset * target_fs)
    offset_sample = int(offset * target_fs)
    segment_len = target_fs  # 1秒

    # 使用生成器逐个产生段落数据，避免在内存中存储所有数据
    def generate_segments():
        # 常规1秒滑窗
        for i in range(0, channel_data_resampled.shape[1], segment_len):
            segment = channel_data_resampled[:, i:i+segment_len]
            if segment.shape[1] == segment_len:
                # 重塑为[19, 1, 200]格式
                segment_reshaped = segment.reshape(19, 1, 200)
                
                if seizure_present and not (i+segment_len < onset_sample or i > offset_sample):
                    y = 1
                else:
                    y = 0
                yield segment_reshaped, y, rec_id
        
        # 正样本增强（0.5秒滑窗）
        if seizure_present:
            for start in range(max(0, onset_sample-target_fs), 
                             min(offset_sample+target_fs, channel_data_resampled.shape[1]-segment_len+1), 
                             int(0.5*target_fs)):
                segment = channel_data_resampled[:, start: start+segment_len]
                if segment.shape[1] == segment_len:
                    segment_reshaped = segment.reshape(19, 1, 200)
                    yield segment_reshaped, 1, rec_id
    
    return generate_segments()

def save_batch_data(batch_data, batch_labels, batch_ids, subfolder, batch_idx):
    """保存批量数据为单个文件"""
    out_dir = os.path.join(output_folder, subfolder)
    batch_array = np.stack(batch_data, axis=0)  # [batch_size, 19, 1, 200]
    labels_array = np.array(batch_labels)
    
    save_name = f"batch_{batch_idx}.pkl"
    save_path = os.path.join(out_dir, save_name)
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            "X": batch_array, 
            "y": labels_array, 
            "fs": 200,
            "record_ids": batch_ids
        }, f)
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"保存失败，文件不存在: {save_path}")
    
    return len(batch_data)

def process_record_wrapper(rec_id, label):
    """处理单个记录的包装函数"""
    mat_file = os.path.join(input_folder, rec_id + '.mat')
    if not os.path.exists(mat_file):
        print(f"{mat_file} 不存在，跳过。")
        return None
    
    try:
        eeg_data = sio.loadmat(mat_file, simplify_cells=True)
        ch_names = [x.strip(' ') for x in eeg_data.get('channels')]
        data = eeg_data.get('data')
        fs = eeg_data.get('fs')
        return process_one_record(rec_id, label, ch_names, data, fs)
    except Exception as e:
        print(f"处理记录 {rec_id} 时出错: {e}")
        return None

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

    def get_partial(refs):
        n = int(len(refs) * PROCESS_RATIO)
        return refs[:n]

    train_refs = get_partial(train_refs)
    val_refs = get_partial(val_refs)
    test_refs = get_partial(test_refs)

    total_saved = 0
    batch_idx = 0

    for subset_name, subset_refs in [("train", train_refs), ("val", val_refs), ("test", test_refs)]:
        print(f"处理 {subset_name} 数据集...")

        # 批次数据缓存
        batch_data = []
        batch_labels = []
        batch_ids = []
        
        pos_count = 0
        neg_count = 0

        # 单线程逐个处理记录
        for rec_id, label in tqdm(subset_refs, desc=f"处理 {subset_name} 记录"):
            segment_generator = process_record_wrapper(rec_id, label)
            
            if segment_generator is None:
                continue
                
            # 逐个处理生成的段落
            for seg_data, seg_label, record_id in segment_generator:
                batch_data.append(seg_data)
                batch_labels.append(seg_label)
                batch_ids.append(record_id)
                
                # 统计正负样本
                if seg_label == 1:
                    pos_count += 1
                else:
                    neg_count += 1
                
                # 当批次满时保存并清空缓存
                if len(batch_data) >= batch_size:
                    saved_count = save_batch_data(batch_data, batch_labels, batch_ids, subset_name, batch_idx)
                    total_saved += saved_count
                    batch_idx += 1
                    
                    # 清空批次缓存以释放内存
                    batch_data.clear()
                    batch_labels.clear()
                    batch_ids.clear()

        # 保存最后不满一个批次的数据
        if len(batch_data) > 0:
            saved_count = save_batch_data(batch_data, batch_labels, batch_ids, subset_name, batch_idx)
            total_saved += saved_count
            batch_idx += 1
            
            # 清空缓存
            batch_data.clear()
            batch_labels.clear()
            batch_ids.clear()

        print(f"{subset_name} 数据集正样本数量: {pos_count}, 负样本数量: {neg_count}")
        print(f"{subset_name} 数据集处理完成")

    print(f"全部分割、保存批量pkl完成。总共保存 {total_saved} 个样本，{batch_idx} 个批次文件")