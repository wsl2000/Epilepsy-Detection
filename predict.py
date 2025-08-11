# -*- coding: utf-8 -*-
"""
推理脚本：载入 CBraMod 模型
使用原始 EEG 数据，不使用 3-montage 预处理
"""

from typing import List, Dict, Any
import json
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import os
import sys

# 添加 CBraMod 相关路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'CBraMod'))

try:
    from CBraMod.models.model_for_wike25 import Model
except ImportError:
    print("警告: 无法导入 CBraMod 模型，请确保路径正确")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 目标通道顺序 - 与训练时保持一致
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4',
    'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]

class MockParams:
    """模拟训练时的参数类"""
    def __init__(self, params_dict):
        for key, value in params_dict.items():
            setattr(self, key, value)

def normalize_per_channel(segment):
    """每通道归一化"""
    mean = segment.mean(axis=1, keepdims=True)
    std = segment.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (segment - mean) / std

def scale_per_channel(segment_norm, amp=100.0):
    """每通道缩放到指定幅度"""
    seg = segment_norm.copy()
    for i in range(seg.shape[0]):
        max_v = np.max(np.abs(seg[i]))
        if max_v > 1e-6:
            seg[i] = seg[i] * (amp / max_v)
    return seg

def create_patches(data, patch_size=200, overlap=0):
    """
    将EEG数据分割成patches
    data: [channels, time_points]
    return: [channels, num_patches, patch_size]
    """
    channels, time_points = data.shape
    step = patch_size - overlap
    
    if time_points < patch_size:
        # 如果数据太短，进行零填充
        padded_data = np.zeros((channels, patch_size))
        padded_data[:, :time_points] = data
        return padded_data.reshape(channels, 1, patch_size)
    
    num_patches = (time_points - patch_size) // step + 1
    patches = np.zeros((channels, num_patches, patch_size))
    
    for i in range(num_patches):
        start_idx = i * step
        patches[:, i, :] = data[:, start_idx:start_idx + patch_size]
    
    return patches

def create_patches(data, patch_size=200, target_patches=10):
    """
    将EEG数据分割成固定数量的patches
    data: [channels, time_points] 
    return: [channels, num_patches, patch_size]
    """
    channels, time_points = data.shape
    
    # 如果数据长度不够，进行填充
    if time_points < target_patches * patch_size:
        pad_length = target_patches * patch_size - time_points
        data = np.pad(data, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
        time_points = data.shape[1]
    
    # 如果数据太长，截取前面部分
    elif time_points > target_patches * patch_size:
        data = data[:, :target_patches * patch_size]
        time_points = data.shape[1]
    
    # 重塑为patches
    patches = data.reshape(channels, target_patches, patch_size)
    
    return patches

def preprocess_eeg_data(channels, data, fs, target_fs=200):
    """
    预处理 EEG 数据以匹配 CBraMod 的输入格式
    返回格式: [channels, num_patches, patch_size] - 与训练时一致
    """
    # 重采样到目标采样率
    if fs != target_fs:
        data = signal.resample_poly(data, target_fs, int(fs), axis=1)
    
    # 重排通道到目标通道顺序
    channel_data = []
    for ch in TARGET_CHANNELS:
        if ch in channels:
            idx = channels.index(ch)
            channel_data.append(data[idx])
        else:
            # 如果通道不存在，用零填充
            channel_data.append(np.zeros(data.shape[1], dtype=np.float32))
    
    channel_data = np.stack(channel_data, axis=0)
    
    # 每通道归一化和缩放 - 与训练时保持一致
    channel_data = normalize_per_channel(channel_data)
    channel_data = scale_per_channel(channel_data, amp=100.0)
    
    # 创建patches: 10秒数据(2500点) -> 10个patches，每个250点
    # 但根据模型期望的38000 = 19×10×200，应该是200点每个patch
    patches = create_patches(channel_data, patch_size=200, target_patches=10)
    
    return patches

def predict_labels(channels: List[str], data: np.ndarray,
                   fs: float, reference_system: str,
                   model_name: str = "model.json") -> Dict[str, Any]:
    """
    CBraMod 预测接口函数 - 修改版
    首先判断最大prob是否大于0.06，然后使用0.0025阈值确定onset/offset
    """
    
    if not hasattr(predict_labels, "count"):
        predict_labels.count = 0
    predict_labels.count += 1
    
    # 1. 读取元数据 & 模型参数
    with open(model_name, "r") as f:
        params_dict = json.load(f)

    # 创建模拟参数对象
    params = MockParams(params_dict)
    
    # 2. 加载 CBraMod 模型
    try:
        model = Model(params).to(DEVICE).eval()
        
        # 加载训练好的权重
        state_dict = torch.load(params_dict["model_weight_path"], map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return {"seizure_present": False,
                "seizure_confidence": 0.,
                "onset": -1,
                "onset_confidence": 0.,
                "offset": -1,
                "offset_confidence": 0.}
    
    # 获取预测参数
    prob_th = params_dict.get("prob_th", 0.5)
    win_sec = params_dict.get("win_sec", 10)  # 10秒窗口
    tgt_fs = params_dict.get("fs", 200)  # 目标采样率，通常是200Hz
    
    # 保存原始采样率 - 用于时间转换
    original_fs = fs

    # print(f"第{predict_labels.count}轮检查，数据长度为{data.shape[1]}个采样点，原始采样率为{fs}Hz")
    # 3. 预处理数据 - 重采样
    if fs != tgt_fs:
        data = signal.resample_poly(data, tgt_fs, int(fs), axis=1)
    
    total_samples = data.shape[1]
    win_samp = int(win_sec * tgt_fs)  # 10秒 = 2000个采样点
    
    # ====== 检测阶段 ======
    print(f"正在进行癫痫检测..., win_size={win_samp}")
    
    # 不重叠的10秒窗口数量
    n_seg = total_samples // win_samp
    
    if n_seg == 0:  # 数据太短
        return {"seizure_present": False,
                "seizure_confidence": 0.,
                "onset": -1,
                "onset_confidence": 0.,
                "offset": -1,
                "offset_confidence": 0.}
    
    # 计算所有窗口的概率
    window_probs = []
    window_times = []
    
    with torch.no_grad():
        for i in range(n_seg):
            start_idx = i * win_samp
            start_time = i * win_sec
            window_times.append(start_time)
            
            # 提取10秒片段
            segment_data = data[:, start_idx:start_idx + win_samp]
            
            # 预处理当前片段
            processed_patches = preprocess_eeg_data(channels, segment_data, tgt_fs, tgt_fs)
            
            # 转换为模型输入格式
            segment_tensor = torch.tensor(processed_patches, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # 模型预测
            output = model(segment_tensor)
            
            # 计算概率
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
                
            prob = torch.sigmoid(logits).cpu().item()
            window_probs.append(prob)
            # print(f"  窗口 {i+1}/{n_seg}: 概率={prob:.4f}, 时间={start_time:.1f}s")

    window_probs = np.array(window_probs)
    window_times = np.array(window_times)
    
    # 获取最大概率值
    max_prob = window_probs.max()
    
    # 判断是否有癫痫发作 - 使用0.99阈值
    if max_prob < 0.979:
        print(f"第{predict_labels.count}轮检查：最大概率 {max_prob:.4f} <= 0.979，判定为无癫痫")
        return {"seizure_present": False,
                "seizure_confidence": float(max_prob),
                "onset": -1,
                "onset_confidence": 0.,
                "offset": -1,
                "offset_confidence": 0.}
    
    # 找出概率大于0.0025的窗口
    low_th_candidates = np.where(window_probs > 0.7)[0]
    
    if len(low_th_candidates) == 0:  # 理论上不应该发生，因为max_prob > 0.06
        print(f"第{predict_labels.count}轮检查：异常情况，无窗口大于0.0025阈值")
        return {"seizure_present": False,
                "seizure_confidence": float(max_prob),
                "onset": -1,
                "onset_confidence": 0.,
                "offset": -1,
                "offset_confidence": 0.}
    
    # 取第一个大于0.0025的窗口的结束点作为onset
    first_window_idx = low_th_candidates[0]
    onset_sec = window_times[first_window_idx] + win_sec/2  # 窗口后端
    
    # 取最后一个大于0.0025的窗口的开始点作为offset
    last_window_idx = low_th_candidates[-1]
    offset_sec = window_times[last_window_idx]  # 窗口前端
    
    # 置信度计算
    onset_confidence = float(window_probs[first_window_idx])
    offset_confidence = float(window_probs[last_window_idx])
    
    print(f"第{predict_labels.count}轮检查：检测完成")
    print(f"  最大概率: {max_prob:.4f} > 0.99，判定为有癫痫")
    print(f"  找到 {len(low_th_candidates)} 个大于0.8阈值的窗口")
    print(f"  主要片段: Onset={onset_sec:.1f}s (窗口{first_window_idx+1}后端), Offset={offset_sec:.1f}s (窗口{last_window_idx+1}前端)")
    
    return {"seizure_present": True,
            "seizure_confidence": float(max_prob),
            "onset": onset_sec,
            "onset_confidence": onset_confidence,
            "offset": offset_sec,
            "offset_confidence": offset_confidence}


if __name__ == "__main__":
    # 测试代码
    print("CBraMod 预测模块已加载")
    print(f"使用设备: {DEVICE}")
    print(f"目标通道数: {len(TARGET_CHANNELS)}")