import pickle
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# 源数据文件夹（已处理过的CHB-MIT）
root = "D:\\datasets\\eeg\\dataset_processed\\CHB-MIT"
# 输出分割后数据的文件夹
out = "D:\\datasets\\eeg\\dataset_processed\\CHB-MIT_seg"

# 若输出目录不存在，则创建
if not os.path.exists(out):
    os.makedirs(out)

# 数据集划分：指定哪些病人为test/val/train
test_pats = ["chb23", "chb24"]
val_pats = ["chb21", "chb22"]
train_pats = [
    "chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
    "chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb19", "chb20",
]
# 统一处理的通道顺序
channels = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]
SAMPLING_RATE = 256  # 采样率

def sub_to_segments(folder, out_folder):
    """
    读取每个病人的全部pkl文件，将每个文件分割为10秒一段的小片段，并标注是否包含癫痫发作。
    :param folder: 病人文件夹名
    :param out_folder: 输出的类别文件夹路径（train/val/test）
    """
    print(f"Processing {folder}...")
    # 遍历该病人所有数据文件
    for f in tqdm(os.listdir(os.path.join(root, folder))):
        print(f"Processing {folder}/{f}...")
        record = pickle.load(open(os.path.join(root, folder, f), "rb"))
        # record为dict，每个通道一个key，record['metadata']含癫痫发作信息

        # 拼接channels顺序，得到 [通道数, 时间点数] 的数据
        signal = []
        for channel in channels:
            if channel in record:
                signal.append(record[channel])
            else:
                raise ValueError(f"Channel {channel} not found in record {record}")
        signal = np.array(signal)

        # 获取癫痫发作的起止时间（采样点下标）
        if "times" in record["metadata"]:
            seizure_times = record["metadata"]["times"]
        else:
            seizure_times = []

        # 将信号按10秒一段滑窗分割，判断该段内是否含癫痫发作
        for i in range(0, signal.shape[1], SAMPLING_RATE * 10):
            segment = signal[:, i : i + 10 * SAMPLING_RATE]
            if segment.shape[1] == 10 * SAMPLING_RATE:
                label = 0  # 默认无癫痫发作

                # 检查此段是否包含发作（起止有交集即算）
                for seizure_time in seizure_times:
                    if (
                        i < seizure_time[0] < i + 10 * SAMPLING_RATE
                        or i < seizure_time[1] < i + 10 * SAMPLING_RATE
                    ):
                        label = 1
                        break

                # 保存分割片段及标签
                pickle.dump(
                    {"X": segment, "y": label},
                    open(
                        os.path.join(out_folder, f"{f.split('.')[0]}-{i}.pkl"),
                        "wb",
                    ),
                )

        # 针对癫痫发作段，进一步以步长5秒密集采样，增强正样本
        for idx, seizure_time in enumerate(seizure_times):
            for i in range(
                max(0, seizure_time[0] - SAMPLING_RATE),
                min(seizure_time[1] + SAMPLING_RATE, signal.shape[1]),
                5 * SAMPLING_RATE,
            ):
                segment = signal[:, i : i + 10 * SAMPLING_RATE]
                label = 1
                # 保存增强片段
                pickle.dump(
                    {"X": segment, "y": label},
                    open(
                        os.path.join(
                            out_folder, f"{f.split('.')[0]}-s-{idx}-add-{i}.pkl"
                        ),
                        "wb",
                    ),
                )

# 整理每个病人的输出文件夹路径（train/val/test）
folders = os.listdir(root)
out_folders = []
for folder in folders:
    if folder in test_pats:
        out_folder = os.path.join(out, "test")
    elif folder in val_pats:
        out_folder = os.path.join(out, "val")
    else:
        out_folder = os.path.join(out, "train")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_folders.append(out_folder)

if __name__ == '__main__':
    # 多进程并行处理所有病人文件夹
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.starmap(sub_to_segments, zip(folders, out_folders))