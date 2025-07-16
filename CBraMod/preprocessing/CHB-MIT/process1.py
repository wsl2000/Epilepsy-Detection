import os
from collections import defaultdict
import pyedflib
import pyedflib.highlevel as hl
import numpy as np
import copy
import shutil
import bz2
import pickle
import _pickle as cPickle
import multiprocessing as mp

# 对数据进行pickle序列化并压缩保存
def compressed_pickle(title, data):
    """
    序列化并压缩数据到文件
    :param title: 保存的文件名
    :param data: 要保存的数据
    """
    # with bz2.BZ2File(title + '.pbz2', 'w') as f:
    #     cPickle.dump(data, f)
    pickle.dump(data, open(title, "wb"))

# 处理metadata摘要文件，提取发作次数、发作起止点等信息
def process_metadata(summary, filename):
    """
    从summary文件中提取指定EDF文件的癫痫发作相关元数据
    :param summary: summary文件路径
    :param filename: 目标EDF文件名
    :return: 包含发作信息的字典
    """
    f = open(summary, "r")

    metadata = {}
    lines = f.readlines()
    times = []
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3 and line[2] == filename:
            j = i + 1
            processed = False
            while not processed:
                if lines[j].split()[0] == "Number":
                    seizures = int(lines[j].split()[-1])
                    processed = True
                j = j + 1

            # 如果有癫痫发作，提取每次发作的起止采样点
            if seizures > 0:
                j = i + 1
                for s in range(seizures):
                    # 保存每次发作的起止点
                    processed = False
                    while not processed:
                        l = lines[j].split()
                        # print(l)

                        if l[0] == "Seizure" and "Start" in l:
                            start = int(l[-2]) * 256 - 1  # 起始点
                            end = (
                                int(lines[j + 1].split()[-2]) * 256 - 1
                            )  # 结束点
                            processed = True
                        j = j + 1
                    times.append((start, end))

            metadata["seizures"] = seizures
            metadata["times"] = times

    return metadata

# 读取EDF文件，仅保留指定通道
def drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None):
    """
    只保留需要的通道信号
    :param edf_source: EDF源文件路径
    :param edf_target: 目标文件路径（未用）
    :param to_keep: 要保留的通道索引列表
    :return: {channel_name: signal_array}
    """
    signals, signal_headers, header = hl.read_edf(
        edf_source, ch_nrs=to_keep, digital=False
    )
    clean_file = {}
    for signal, header in zip(signals, signal_headers):
        channel = header.get("label")
        if channel in clean_file.keys():
            channel = channel + "-2"
        clean_file[channel] = signal
    return clean_file

# 保留目标通道，缺失通道补零，然后保存
def move_channels(clean_dict, channels, target):
    """
    处理通道，只保留目标通道，不存在的补零，最终保存
    :param clean_dict: {channel: signal}
    :param channels: 目标通道字典
    :param target: 输出文件路径
    """
    # 标记需要删除的无关通道
    keys_to_delete = []
    for key in clean_dict:
        if key != "metadata" and key not in channels.keys():
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del clean_dict[key]

    # 获取信号长度
    size = 0
    for item in clean_dict.keys():
        if item != "metadata":
            size = len(clean_dict.get(item))
            break

    # 缺失通道补零
    for k in channels.keys():
        if k not in clean_dict.keys():
            clean_dict[k] = np.zeros(size, dtype=float)

    compressed_pickle(target + ".pkl", clean_dict)

# 批量处理指定病人的EDF文件
def process_files(pacient, valid_channels, channels, start, end):
    """
    按编号批量处理某位病人的EDF文件
    :param pacient: 病人编号
    :param valid_channels: 参考通道列表
    :param channels: 通道字典
    :param start: 起始文件编号
    :param end: 结束文件编号
    """
    for num in range(start, end + 1):
        to_keep = []

        num = ("0" + str(num))[-2:]
        filename = "{path}/chb{p}/chb{p}_{n}.edf".format(
            path=signals_path, p=pacient, n=num
        )

        # 检查通道，确定实际可用通道索引
        try:
            signals, signal_headers, header = hl.read_edf(filename, digital=False)
            n = 0
            for h in signal_headers:
                if h.get("label") in valid_channels:
                    if n not in to_keep:
                        to_keep.append(n)
                n = n + 1

        except OSError:
            print("****************************************")
            print("WARNING - Do not worry")
            print("File", filename, "does not exist.\nProcessing next file.")
            print("****************************************")
            continue

        if len(to_keep) > 0:
            try:
                print(
                    "Removing",
                    len(signal_headers) - len(to_keep),
                    "channels from file ",
                    "chb{p}_{n}.edf".format(p=pacient, n=num),
                )
                clean_dict = drop_channels(
                    filename,
                    edf_target="{path}/chb{p}/chb{p}_{n}.edf".format(
                        path=clean_path, p=pacient, n=num
                    ),
                    to_keep=to_keep,
                )
                print("Processing file ", filename)
            except AssertionError:
                print("****************************************")
                print("WARNING - Do not worry")
                print("File", filename, "does not exist.\nProcessing next file.")
                print("****************************************")
                continue

        # 处理元数据
        metadata = process_metadata(
            "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient),
            "chb{p}_{n}.edf".format(p=pacient, n=num),
        )
        metadata["channels"] = valid_channels
        clean_dict["metadata"] = metadata
        target = "{path}/chb{p}/chb{p}_{n}.edf".format(
            path=clean_path, p=pacient, n=num
        )
        move_channels(clean_dict, channels, target)

# 处理某个病人某一参考文件，并批量处理剩余文件
def start_process(pacient, num, start, end, sum_ind):
    """
    处理单个病人：先处理参考文件，提取通道，再批量处理其余文件
    :param pacient: 病人编号
    :param num: 参考文件编号
    :param start: 需处理的起始文件编号
    :param end: 需处理的结束文件编号
    :param sum_ind: summary索引（用于通道变更情况）
    """
    # 打开summary文件，初始化通道相关变量
    f = open(
        "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient), "r"
    )

    channels = defaultdict(list)  # 通道名到索引的映射
    valid_channels = []  # 有效通道名列表
    to_keep = []  # 需保留的通道索引

    channel_index = 1  # 通道索引计数器
    summary_index = 0  # summary中通道变更的块索引

    # 遍历summary，提取通道信息
    for line in f:
        line = line.split()
        if len(line) == 0:
            continue

        if line[0] == "Channels" and line[1] == "changed:":
            summary_index += 1

        if (
            line[0] == "Channel"
            and summary_index == sum_ind
            and (line[2] != "-" and line[2] != ".")
        ):  # "-"表示无效通道
            if (
                line[2] in channels.keys()
            ):  # 通道名重复，后缀加-2
                name = line[2] + "-2"
            else:
                name = line[2]

            # 添加通道到字典和列表
            channels[name].append(str(channel_index))
            channel_index += 1
            valid_channels.append(name)
            to_keep.append(int(line[1][:-1]) - 1)

    # 处理参考EDF文件
    filename = "{path}/chb{p}/chb{p}_{n}.edf".format(
        path=signals_path, p=pacient, n=num
    )
    target = "{path}/chb{p}/chb{p}_{n}.edf".format(path=clean_path, p=pacient, n=num)

    # 若输出目录不存在则新建
    if not os.path.exists("{path}/chb{p}".format(p=pacient, path=clean_path)):
        os.makedirs("{path}/chb{p}".format(p=pacient, path=clean_path))

    clean_dict = drop_channels(filename, edf_target=target, to_keep=to_keep)

    # 处理元数据
    metadata = process_metadata(
        "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient),
        "chb{p}_{n}.edf".format(p=pacient, n=num),
    )

    metadata["channels"] = valid_channels
    clean_dict["metadata"] = metadata

    compressed_pickle(target + ".pkl", clean_dict)

    # 批量处理其它EDF文件
    process_files(pacient, valid_channels, channels, start, end)

# ======================== 全局参数 ========================
signals_path = r"D:\\datasets\\eeg\\dataset_dir_original\\chb-mit-scalp-eeg-database-1.0.0"  # 原始数据主目录
clean_path = r"D:\\datasets\\eeg\\dataset_processed\\CHB-MIT"  # 清洗后数据存放目录

if not os.path.exists(clean_path):
    os.makedirs(clean_path)

# 指定每位病人的处理参数
parameters = [
    ("01", "01", 2, 46, 0),
    ("02", "01", 2, 35, 0),
    ("03", "01", 2, 38, 0),
    ("05", "01", 2, 39, 0),
    ("06", "01", 2, 24, 0),
    ("07", "01", 2, 19, 0),
    ("08", "02", 3, 29, 0),
    ("10", "01", 2, 89, 0),
    ("11", "01", 2, 99, 0),
    ("14", "01", 2, 42, 0),
    ("20", "01", 2, 68, 0),
    ("21", "01", 2, 33, 0),
    ("22", "01", 2, 77, 0),
    ("23", "06", 7, 20, 0),
    ("24", "01", 3, 21, 0),
    ("04", "07", 1, 43, 1),
    ("09", "02", 1, 19, 1),
    ("15", "02", 1, 63, 1),
    ("16", "01", 2, 19, 0),
    ("18", "02", 1, 36, 1),
    ("19", "02", 1, 30, 1),
]
# parameters = [
#     ("12", "")
# ]

# ======================== 主入口 ========================
if __name__ == '__main__':
    # 多进程并发处理全部病人
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.starmap(start_process, parameters)