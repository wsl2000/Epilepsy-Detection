import os
import pickle
import numpy as np
import csv
import random
import scipy.io as sio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

input_folder = r"/work/projects/project02629/datasets/dataset_dir_original/shared_data/training"
output_folder = r"/work/projects/project02629/datasets/processed/wike25_original_channels"

# 创建输出文件夹
for sub in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

def load_all_references(folder: str):
    """加载所有参考标签"""
    refs = []
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rec_id = row[0]
            label = (int(row[1]), float(row[2]), float(row[3]))
            refs.append((rec_id, label))
    return refs

def process_one_record(rec_id, label, ch_names, data, fs, subfolder):
    """
    处理单个记录，保留原始通道和原始采样率
    """
    out_dir = os.path.join(output_folder, subfolder)
    seizure_present, onset, offset = label
    
    # 直接使用原始数据，不进行双极导联转换
    original_data = data
    num_channels = original_data.shape[0]
    
    print(f"记录 {rec_id}: {num_channels} 个通道, 原始采样率: {fs} Hz")
    
    # 10秒窗口长度（使用原始采样率）
    win_len_samples = int(10 * fs)  # 10秒对应的原始采样点数
    
    # onset/offset对应的采样点
    onset_sample = int(onset * fs) if onset > 0 else -1
    offset_sample = int(offset * fs) if offset > 0 else -1
    
    # 常规分割：不重叠的10秒窗口
    for i in range(0, original_data.shape[1], win_len_samples):
        segment = original_data[:, i:i+win_len_samples]
        
        if segment.shape[1] == win_len_samples:
            # 判断标签：窗口是否包含癫痫发作
            label_val = 0
            if seizure_present and onset_sample >= 0 and offset_sample >= 0:
                if (i < onset_sample < i + win_len_samples or 
                    i < offset_sample < i + win_len_samples or
                    (onset_sample <= i and offset_sample >= i + win_len_samples)):
                    label_val = 1
            
            # 保存片段，包含通道名称信息和原始采样率
            pickle.dump(
                {
                    "X": segment,  # 保存原始长度的数据
                    "y": label_val, 
                    "channels": ch_names, 
                    "num_channels": num_channels,
                    "fs": fs  # 保存原始采样率信息
                },
                open(os.path.join(out_dir, f"{rec_id}-{i}.pkl"), "wb")
            )
    
    # 数据增强：癫痫片段增强
    if seizure_present and onset_sample >= 0 and offset_sample >= 0:
        # 在癫痫发作区域附近生成更多样本
        for idx, i in enumerate(range(
            max(0, onset_sample - int(fs)),  # 发作前1秒开始
            min(offset_sample + int(fs), original_data.shape[1]),  # 发作后1秒结束
            int(5 * fs)  # 每5秒取一个窗口
        )):
            segment = original_data[:, i:i+win_len_samples]
            
            if segment.shape[1] == win_len_samples:
                pickle.dump(
                    {
                        "X": segment,
                        "y": 1, 
                        "channels": ch_names, 
                        "num_channels": num_channels,
                        "fs": fs
                    },
                    open(os.path.join(out_dir, f"{rec_id}-s-{idx}-add-{i}.pkl"), "wb")
                )
            elif segment.shape[1] > 0:
                # 处理不足10秒的边界情况：用最后一个值填充到10秒
                padded_segment = np.zeros((segment.shape[0], win_len_samples), dtype=np.float32)
                padded_segment[:, :segment.shape[1]] = segment
                
                # 用每个通道最后一个值填充剩余部分
                for ch_idx in range(segment.shape[0]):
                    if segment.shape[1] > 0:
                        last_value = segment[ch_idx, -1]
                        padded_segment[ch_idx, segment.shape[1]:] = last_value
                
                pickle.dump(
                    {
                        "X": padded_segment,
                        "y": 1, 
                        "channels": ch_names, 
                        "num_channels": num_channels,
                        "fs": fs
                    },
                    open(os.path.join(out_dir, f"{rec_id}-s-{idx}-add-{i}.pkl"), "wb")
                )

def process_record_wrapper(args):
    """处理记录的包装函数"""
    rec_id, label, subfolder = args
    mat_file = os.path.join(input_folder, rec_id + '.mat')
    
    if not os.path.exists(mat_file):
        print(f"警告: MAT文件不存在: {mat_file}")
        return
    
    try:
        # 读取MAT文件
        eeg_data = sio.loadmat(mat_file, simplify_cells=True)
        ch_names = [x.strip() for x in eeg_data.get('channels')]
        data = eeg_data.get('data')
        fs = eeg_data.get('fs')
        
        # 处理单个记录
        process_one_record(rec_id, label, ch_names, data, fs, subfolder)
        
    except Exception as e:
        print(f"处理记录 {rec_id} 时出错: {e}")

def main():
    """主函数"""
    random.seed(42)
    
    # 加载所有参考标签
    refs = load_all_references(input_folder)
    random.shuffle(refs)
    
    # 数据集分割（8:1:1）
    N = len(refs)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    
    train_refs = refs[:n_train]
    val_refs = refs[n_train:n_train+n_val]
    test_refs = refs[n_train+n_val:]
    
    print(f"数据集分割: 训练={len(train_refs)}, 验证={len(val_refs)}, 测试={len(test_refs)}")
    
    # 准备所有处理任务
    all_args = []
    for rec_id, label in train_refs:
        all_args.append((rec_id, label, "train"))
    for rec_id, label in val_refs:
        all_args.append((rec_id, label, "val"))
    for rec_id, label in test_refs:
        all_args.append((rec_id, label, "test"))
    
    # 并行处理
    print("开始处理数据...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        tasks = [executor.submit(process_record_wrapper, args) for args in all_args]
        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="处理进度"):
            pass
    
    print("数据预处理完成！")
    
    # 统计结果
    for subset in ["train", "val", "test"]:
        subset_dir = os.path.join(output_folder, subset)
        if os.path.exists(subset_dir):
            n_files = len([f for f in os.listdir(subset_dir) if f.endswith('.pkl')])
            n_seizure = len([f for f in os.listdir(subset_dir) if "add" in f])
            n_normal = n_files - n_seizure
            print(f"{subset}集: {n_files} 个文件, 癫痫={n_seizure}, 正常={n_normal}")

if __name__ == "__main__":
    main()