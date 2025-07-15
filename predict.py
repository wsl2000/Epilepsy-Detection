# -*- coding: utf-8 -*-
"""

脚本用于测试预训练模型

@author:  Maurice Rohr, Dirk Schweickard
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from wettbewerb import get_3montages

# 课程示例中用到的包
import mne
from scipy import signal as sig
import ruptures as rpt

### 方法签名（参数和返回值数量）不能更改
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model.json') -> Dict[str,Any]:
    '''
    参数
    ----------
    channels : List[str]
        提供的通道名称
    data : ndarray
        指定通道的EEG信号
    fs : float
        信号的采样频率
    reference_system :  str
        使用的参考系统，“Bezugselektrode”，不保证正确！
    model_name : str
        你们的模型名称，例如在提交时指定的名称。
        可用于从文件夹中加载正确的模型
    返回
    -------
    prediction : Dict[str,Any]
        包含预测结果，是否有癫痫发作以及如果有，发作的起止时间（Onset+Offset）
    '''

#------------------------------------------------------------------------------
# 你们的代码从这里开始

    # 初始化返回结果（结果）
    seizure_present = True # 是否存在癫痫发作
    seizure_confidence = 0.5 # 模型的不确定性（可选）
    onset = 4.2   # 癫痫发作开始时间（秒）
    onset_confidence = 0.99 # 对开始时间的不确定性（可选）
    offset = 999999  # 癫痫发作结束时间（可选）
    offset_confidence = 0   # 对结束时间的不确定性（可选）

    # 这里可以加载你们的预训练模型（也可以由多个文件组成）
    with open(model_name, 'rb') as f:  
        parameters = json.load(f)         # 加载简单模型（1个参数）
        th_opt = parameters['std_thresh']

    # 应用课程中的示例代码

    _montage, _montage_data, _is_missing = get_3montages(channels, data)
    signal_std = np.zeros(len(_montage))
    for j, signal_name in enumerate(_montage):
        # 取EEG的第一个导联
        signal = _montage_data[j]
        # 应用陷波滤波器以抑制电网频率
        signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
        # 应用0.5Hz到70Hz的带通滤波器以滤除噪声
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
        
        # 计算信号的短时傅里叶变换：signal_filtered = 通道的滤波信号, fs = 采样频率, nperseg = 每段长度
        # 输出 f= 采样频率数组, t = 段时间数组, Zxx = 信号的STFT
        f, t, Zxx = sig.stft(signal_filter, fs, nperseg=fs * 3)
        # 计算频率步长
        df = f[1] - f[0]
        # 基于STFT的实部和虚部计算能量（幅值）
        E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df
        
        signal_std[j] = np.std(signal_filter)

        # 每个病人第一次循环时创建新数组
        if j == 0:
            # 用第一个通道的能量信号初始化数组
            E_array = np.array(E_Zxx)
        else:
            # 将新的能量信号添加到已有通道（堆叠）
            E_array = np.vstack((E_array, np.array(E_Zxx)))
            
    # 计算用于癫痫检测的特征
    signal_std_max = signal_std.max()
    # 信号分类
    seizure_present = signal_std_max>th_opt
    
    # 计算每个时间点所有通道的总能量
    E_total = np.sum(E_array, axis=0)
    # 计算最大能量的位置
    max_index = E_total.argmax()

    # 计算总能量的“变化点”
    # 如果最大值在信号开始处，则onset也在开始处，无法计算“变化点”
    if max_index == 0:
        onset = 0.0
        onset_confidence = 0.2
        
    else:
        # 用ruptures包计算“变化点”
        # 设置“线性惩罚分段方法”以检测信号中的变化点，使用rbf代价函数
        algo = rpt.Pelt(model="rbf").fit(E_total)
        # 计算变化点的有序列表，pen = 惩罚值
        result = algo.predict(pen=10)
        # 索引向前偏移1
        result1 = np.asarray(result) - 1
        # 选择最大值之前的变化点
        result_red = result1[result1 < max_index]
        # 如果至少有一个变化点，则取最接近最大值的那个
        if len(result_red)<1:
            # 如果没有找到变化点，则猜测其“接近”最大值
            print('No changepoint, taking maximum')
            onset_index = max_index
        else:
            # 变化点即为onset
            onset_index = result_red[-1]
        # 返回onset
        onset = t[onset_index]      
     
#------------------------------------------------------------------------------
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # 返回预测结果的字典 - 必须保持不变！
