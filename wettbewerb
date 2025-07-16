# -*- coding: utf-8 -*-
"""
此文件不应被修改，由我们提供并重置。

用于加载和保存文件的函数
"""
__author__ = "Maurice Rohr und Dirk Schweickard"

from typing import List, Tuple, Dict, Any
import csv
import scipy.io as sio
import numpy as np
import os


### 注意！不要修改此函数。
def load_references(folder: str = './shared_data/training', idx: int = 0) -> Tuple[List[str], List[List[str]],
                                                          List[np.ndarray],  List[float],
                                                          List[str], List[Tuple[bool,float,float]]]:
    """
    从 .mat（测量数据）和 .csv（标签）文件中读取参考数据。
    Parameters
    ----------
    folder : str, optional
        训练数据的位置。默认值 '../training'。
    idx : int, optional
        开始加载数据的索引。
        例如 idx=10 意味着将加载数据点 10 到 109（包含）
        如果从索引开始剩余文件少于100个，则只加载这些文件。

    Returns
    -------
    ids : List[str]
        记录ID列表
    channels : List[List[str]]
        每个记录的可用通道列表
    data :  List[ndarray]
        每个记录的数据列表
    sampling_frequencies : List[float]
        采样频率列表。
    reference_systems : List[str]
        参考系统列表。"LE", "AR", "Sz"（附加信息）
    """
    
    # 初始化列表 ids, channels, data, sampling_frequencies, reference_systems 和 eeg_labels
    ids: List[str] = []
    channels: List[List[str]] = []
    data: List[np.ndarray] = []
    sampling_frequencies: List[float] = []
    reference_systems: List[str] = []
    eeg_labels: List[Tuple[bool,float,float]] = []
    
    # 从文件夹创建数据集并用数据填充列表
    dataset = EEGDataset(folder)
    data_to_load = 100  # 加载最多100个记录
    if(len(dataset)-idx < data_to_load):
        end = len(dataset)-idx
    else:
        end = data_to_load
    if(idx >= len(dataset)):
        print("数据集小于提供的索引")
        return None
    for i in range(idx, idx+end):
        ids.append(dataset[i][0])
        channels.append(dataset[i][1])
        data.append(dataset[i][2])
        sampling_frequencies.append(dataset[i][3])
        reference_systems.append(dataset[i][4])
        eeg_labels.append(dataset[i][5])
        
    # 显示加载了多少数据
    print("{}\t 个文件已加载。".format(len(ids)))
    return ids, channels, data, sampling_frequencies, reference_systems, eeg_labels

### 注意！不要修改此类。
class EEGDataset:
    def __init__(self,folder:str) -> None:
        """此类表示一个EEG数据集。
        
        用法：
            用以下方式创建新数据集（不加载所有数据）
            dataset = EEGDataset("../training/")
            len(dataset) # 返回数据集大小
            dataset[0] # 返回数据集中第一个元素，包含 (id, channels, data, sampling_frequency, reference_system, eeg_label)
            it = iter(dataset) # 返回数据集的迭代器，
            next(it) # 返回下一个元素，直到所有数据被获取一次
            for item in dataset: # 遍历整个数据集一次
                (id, channels, data, sampling_frequency, reference_system, eeg_label) = item
                # 计算

        Args:
            folder (str): 包含由.mat文件和REFERENCE.csv文件组成的数据集的文件夹
        """
        assert isinstance(folder, str), "参数folder必须是字符串，但给出了{}".format(type(folder))
        assert os.path.exists(folder), '参数folder不存在！'
        # 初始化ids和labels列表
        self._folder = folder
        self._ids: List[str] = []
        self._eeg_labels: List[Tuple[bool,float,float]] = []
        # 加载references文件
        with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # 遍历每一行
            for row in csv_reader:
                self._ids.append(row[0])
                self._eeg_labels.append((int(row[1]),float(row[2]),float(row[3])))
    
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self,idx) -> Tuple[str, List[str],
                                    np.ndarray,  float,
                                    str, Tuple[bool,float,float]]:
        # 加载Matlab文件
        eeg_data = sio.loadmat(os.path.join(self._folder, self._ids[idx] + '.mat'),simplify_cells=True)
        ch_names = eeg_data.get('channels')
        channels = [x.strip(' ') for x in ch_names] 
        data = eeg_data.get('data')
        sampling_frequency = eeg_data.get('fs')
        reference_system = eeg_data.get('reference_system')
        return (self._ids[idx],channels,data,sampling_frequency,reference_system,self._eeg_labels[idx])
    
    def get_labels(self):
        return self._eeg_labels
    
        
    
        




### 注意！不要修改此函数。
#predictions = {"id":id,"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
#                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
#                   "offset_confidence":offset_confidence}
def save_predictions(predictions: List[Dict[str,Any]], folder: str=None, teamID: str=None) -> None:
    """
    函数将给定的预测保存到名为PREDICTIONS.csv的CSV文件中。
    所有可选预测都用默认值替换。
    Parameters
    ----------
    predictions : List[Dict[str,Any]]
        字典列表。每个字典包含字段"id","seizure_present",
                "seizure_confidence"（可选）,"onset","onset_confidence"（可选）,
                "offset"（可选）,"offset_confidence"（可选）
	folder : str
		预测的保存位置
    Returns
    -------
    None.

    """    
	# 检查参数
    assert isinstance(predictions, list), \
        "参数predictions必须是列表，但给出了{}。".format(type(predictions))
    assert len(predictions) > 0, '参数predictions必须是非空列表。'
    assert isinstance(predictions[0], dict), \
        "列表predictions的元素必须是字典，但给出了{}。".format(type(predictions[0]))
    assert "id" in predictions[0], \
        "预测必须有ID，但字典中没有相应的键"
	
    if folder==None:
        file = "PREDICTIONS.csv"
    else:
        if teamID is None:
            file = os.path.join(folder, "PREDICTIONS.csv")
        else:
            file = os.path.join(folder, "PREDICTIONS"+"_"+teamID+".csv")
    # 检查文件是否已存在，如果是则删除文件
    if os.path.exists(file) and teamID is None:
        os.remove(file)

    with open(file, mode='w', newline='') as predictions_file:

        # 初始化CSV写入器来写入文件
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        # 遍历每个预测
        header=["id","seizure_present","seizure_confidence","onset","onset_confidence","offset","offset_confidence"]
        predictions_writer.writerow(header)
        for prediction in predictions:
            _id = prediction["id"]
            _seizure_present = prediction["seizure_present"]
            _seizure_confidence = prediction.get("seizure_confidence",1.0) 
            _onset = prediction["onset"]
            _onset_confidence = prediction.get("onset_confidence",1.0) 
            _offset = prediction.get("offset",999999.0)
            _offset_confidence = prediction.get("offset_confidence",0.0)
            predictions_writer.writerow([_id,_seizure_present,_seizure_confidence,_onset,_onset_confidence,_offset,_offset_confidence])
        # 输出保存了多少个标签（预测）的信息
        print("{}\t 个标签已写入。".format(len(predictions)))
        

def get_3montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
    """
    函数从给定的导联（蒙太奇）计算3个蒙太奇 Fp1-F3, Fp2-F4, C3-P3
    到同一参考电极。如果没有所有必需的电极，相应的信号将被0替换。
    ----------
    channels : List[str]
        通道名称，例如 Fp1, Cz, C3
	data : ndarray
		通道数据
    Returns
    -------
    montages : List[str]
        蒙太奇名称 ["Fp1-F3", "Fp2-F4", "C3-P3"]
    montage_data : ndarray
        蒙太奇数据
    montage_missing : bool
        如果一个或多个蒙太奇缺失则为1，否则为0

    """   
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([3,m])
    montage_missing = 0
    if '-' in channels:
        try:
            montage_data[0,:] = data[channels.index('Fp1-F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2-F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[2,:] = data[channels.index('C3-P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)

    else:
        try:
            montage_data[0,:] = data[channels.index('Fp1')] - data[channels.index('F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2')] - data[channels.index('F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[2,:] = data[channels.index('C3')] - data[channels.index('P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)


def get_6montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
    """
    函数从给定的导联（蒙太奇）计算6个蒙太奇 Fp1-F3, Fp2-F4, C3-P3, F3-C3, F4-C4, C4-P4
    到同一参考电极。如果没有所有必需的电极，相应的信号将被0替换。
    ----------
    channels : List[str]
        通道名称，例如 Fp1, Cz, C3
	data : ndarray
		通道数据
    Returns
    -------
    montages : List[str]
        蒙太奇名称 ["Fp1-F3", "Fp2-F4", "C3-P3", "F3-C3", "F4-C4", "C4-P4"]
    montage_data : ndarray
        蒙太奇数据
    montage_missing : bool
        如果一个或多个蒙太奇缺失则为1，否则为0

    """  
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([6,m])
    montage_missing = 0
    if '-' in channels:
        try:
            montage_data[0,:] = data[channels.index('Fp1-F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2-F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[2,:] = data[channels.index('C3-P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[3,:] = data[channels.index('F3-C3')]
            montages.append('F3-C3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[4,:] = data[channels.index('F4-C4')]
            montages.append('F4-C4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[5,:] = data[channels.index('C4-P4')]
            montages.append('C4-P4')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)

    else:         
        try:
            montage_data[0,:] = data[channels.index('Fp1')] - data[channels.index('F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2')] - data[channels.index('F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[2,:] = data[channels.index('C3')] - data[channels.index('P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[3,:] = data[channels.index('F3')] - data[channels.index('C3')]
            montages.append('F3-C3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[4,:] = data[channels.index('F4')] - data[channels.index('C4')]
            montages.append('F4-C4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[5,:] = data[channels.index('C4')] - data[channels.index('P4')]
            montages.append('C4-P4')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)