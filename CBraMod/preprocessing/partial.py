import os
import shutil
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

src_root = r'D:\datasets\eeg\dataset_processed\CHB-MIT_seg'
dst_root = r'D:\datasets\eeg\dataset_processed\CHB-MIT_seg_partial'
folders = ['train', 'test', 'val']
percent = 0.1  # 取前10%
max_workers = 8  # 线程数，可根据CPU调整

def copy_item(src_path, dst_path):
    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
    elif os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

for folder in folders:
    src_folder = os.path.join(src_root, folder)
    dst_folder = os.path.join(dst_root, folder)
    os.makedirs(dst_folder, exist_ok=True)

    files = sorted(os.listdir(src_folder))
    n_files = math.ceil(len(files) * percent)
    selected_files = files[:n_files]

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(selected_files), desc=f'Copying {folder}') as pbar:
            futures = []
            for fname in selected_files:
                src_path = os.path.join(src_folder, fname)
                dst_path = os.path.join(dst_folder, fname)
                futures.append(executor.submit(copy_item, src_path, dst_path))
            for future in as_completed(futures):
                pbar.update(1)

print("复制完成。")