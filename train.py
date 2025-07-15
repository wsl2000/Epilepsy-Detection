# -*- coding: utf-8 -*-
"""
示例代码和实验场

"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
import json


### if __name__ == '__main__':  # Windows多进程处理时必需

training_folder  = "./shared_data/training_mini"

# 导入EEG文件、相关通道命名、采样频率(Hz)和名称(通常fs=256 Hz)，以及参考系统
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) 

# 癫痫发作检测 (示例代码在此处保存一个模型)
# 初始化数据数组
feature = []
label = []

for i,_id in enumerate(ids):
    _fs = sampling_frequencies[i]
    _eeg_signals = data[i]
    _eeg_label = eeg_labels[i]
    label.append(_eeg_label[0])
    # 计算montage
    _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)
    signal_std = np.zeros(len(_montage))
    for j, signal_name in enumerate(_montage):
        # 提取EEG的第一个montage
        signal = _montage_data[j]
        # 应用陷波滤波器以抑制工频噪声
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
        # 应用0.5Hz到70Hz的带通滤波器以从信号中过滤噪声
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
        
        signal_std[j] = np.std(signal_filter)

    # 只考虑具有最大标准差的通道
    signal_std_max = signal_std.max()
    feature.append(signal_std_max)

X = np.array(feature)
Y = np.array(label)    
best_f1 = 0
th_opt = 0

# 通过遍历不同阈值，寻找使F1分数最大的最佳阈值（th_opt），用于二分类任务的简单模型训练。
for th in np.arange(X.min(),X.max(),(X.max()-X.min())/1e5):
    pred = X>th
    TP = np.sum((pred==Y) & (Y==1))
    FP = np.sum((pred==1) & (Y==0))
    FN = np.sum((pred==0) & (Y==1))
    F1 = 2*TP/(2*TP+FP+FN)  
    if F1 >best_f1: # 记录最佳阈值
        th_opt = th
        best_f1 = F1
print('最优阈值是', th_opt,' 训练数据上的F1分数为',best_f1)

# 保存模型
model_params = {'std_thresh':th_opt}
with open('model.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, ensure_ascii=False, indent=4)
    print('癫痫发作检测模型已保存!')
        
# 发作起始检测 (示例代码在此处不保存模型，因为没有学习参数)
# 初始化数据数组
onset_list_predict = []
onset_list = []
seizure_id_list = []

for i,_id in enumerate(ids):
    # 1. 遍历每个样本，每个样本有自己的采样频率、信号、标签等。
    _fs = sampling_frequencies[i]
    _eeg_signals = data[i]
    _eeg_label = eeg_labels[i]

    # 2. 只处理有癫痫发作的样本
    if _eeg_label[0]:
        onset_list.append(_eeg_label[1])
        seizure_id_list.append(_id)
        # 3. 计算montage
        _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)

        # 4. 对每个Montage通道
        for j, signal_name in enumerate(_montage):
            # 4. 做信号预处理
            # 提取EEG的第一个montage
            signal = _montage_data[j]
            # 应用陷波滤波器以抑制工频噪声
            signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
            # 应用0.5Hz到70Hz的带通滤波器以从信号中过滤噪声
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)

            # 5. 计算短时傅里叶变换（STFT），得到每个时间段的频谱能量（幅度平方和）。
            # 计算信号的短时傅里叶变换: signal_filtered = 通道的滤波信号, fs = 采样频率, nperseg = 每个段的长度
            # 输出 f = 采样频率数组, t = 段时间数组, Zxx = 信号的STFT
            f, t, Zxx = sig.stft(signal_filter, _fs, nperseg=_fs * 3)
            # 计算频率步长
            df = f[1] - f[0]
            # 基于STFT的实部和虚部计算能量(幅度)
            E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df

            # 6. 汇总所有通道的能量
            # 把所有通道的能量堆叠起来，最后对每个时间点求和，得到总能量序列。
            # 在每个患者的第一次迭代中创建新数组
            if j == 0:
                # 用第一个通道的能量信号初始化数组
                E_array = np.array(E_Zxx)
            else:
                # 将新的能量信号添加到现有通道中(叠加)
                E_array = np.vstack((E_array, np.array(E_Zxx)))
                
        # 计算每个时间点所有通道的总能量
        E_total = np.sum(E_array, axis=0)
        # 7. 找到能量最大的位置
        max_index = E_total.argmax()

        '''
            8. 检测变化点（Change Point Detection）
            用ruptures库的Pelt算法检测能量变化点，找到最接近能量峰值前的变化点，作为预测的癫痫发作起始点。
        '''
        # 计算总能量的"变化点"
        # 如果最大值在信号开始处，则起始也必须在开始处，我们无法计算"变化点"
        if max_index == 0:
            onset_list_predict.append(0.0)
        else:
            # 使用ruptures包计算"变化点"
            # 设置"线性惩罚分割方法"以使用rbf成本函数检测信号中的变化点
            algo = rpt.Pelt(model="rbf").fit(E_total)
            # 计算变化点的排序列表, pen = 惩罚值
            result = algo.predict(pen=10)
            # 索引偏移1
            result1 = np.asarray(result) - 1
            # 选择最大值之前的变化点
            result_red = result1[result1 < max_index]
            # 如果至少有一个变化点，我们取最接近最大值的那个
            if len(result_red)<1:
                # 如果没有找到变化点，我们推测它"接近"最大值
                print('没有变化点，取最大值')
                onset_index = max_index
            else:
                # 变化点正好对应起始点
                onset_index = result_red[-1]
            # 将起始点添加到起始点列表中
            onset_list_predict.append(t[onset_index])

# 计算基于医生标注的计算癫痫起始和真实起始之间的绝对误差
prediction_error = np.abs(np.asarray(onset_list_predict) - np.asarray(onset_list))
print('训练时的平均起始预测误差:', np.mean(prediction_error))

# 绘制每个患者的误差图
plt.figure(1)
plt.scatter(np.arange(1, len(prediction_error)+1),prediction_error)
#plt.hlines(10, 0, len(prediction_error)+1, colors='red')
plt.ylabel('误差(秒)')
plt.xlabel('患者')
plt.show()