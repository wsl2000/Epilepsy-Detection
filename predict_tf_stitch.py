# -*- coding: utf-8 -*-
"""
推理脚本：载入CBraMod-inspired Stitched Transformer模型
复用train_tf_stitch.py中的模型定义，与主办方接口保持一致。
"""

from typing import List, Dict, Any
import json, torch, numpy as np
from scipy import signal
import os
from torch import nn

# 导入训练脚本中的模型和组件
from train_tf_stitch import (
    OptimizedStitchedTransformer,
    StitchedEEGTransformer,
    MultiHeadChannelAttention,
    EfficientPatchEmbedding,
    EfficientCrissCrossLayer,
    add_frequency_features,
    DEVICE, TARGET_FS, WIN_SEC, STEP_SEC, WIN_SAMP, STEP_SAMP
)
from wettbewerb import get_6montages

def load_model_from_metadata(model_name: str):
    """
    从元数据文件加载模型配置和权重
    
    Args:
        model_name: 模型元数据JSON文件路径
        
    Returns:
        model: 加载完成的模型
        params: 模型参数字典
    """
    # 读取元数据
    with open(model_name, "r") as f:
        params = json.load(f)
    
    # 获取模型类型
    model_type = params.get("model_type", "OptimizedStitchedTransformer")
    
    # 从元数据中获取模型参数
    model_scale = params.get("model_scale", "small")
    input_channels = params.get("input_channels", 36)
    patch_size = params.get("patch_size", 16)
    dropout = params.get("dropout", 0.1)
    
    # 根据模型类型创建相应的模型
    if model_type == "OptimizedStitchedTransformer":
        # 使用优化版本的模型（速度快）
        model = OptimizedStitchedTransformer(
            input_channels=input_channels,
            patch_size=patch_size,
            dropout=dropout,
            model_scale=model_scale
        ).to(DEVICE).eval()
    elif model_type == "StitchedEEGTransformer":
        # 使用完整版本的模型（功能全）
        d_model = params.get("d_model", 512)
        num_heads = params.get("num_heads", 8)
        num_layers = params.get("num_layers", 12)
        
        model = StitchedEEGTransformer(
            input_channels=input_channels,
            patch_size=patch_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            model_scale=model_scale
        ).to(DEVICE).eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载模型权重
    model_weight_path = params["model_weight_path"]
    if not os.path.exists(model_weight_path):
        # 如果绝对路径不存在，尝试相对路径
        model_weight_path = os.path.join(os.path.dirname(model_name), 
                                       os.path.basename(model_weight_path))
    
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Model weights not found: {model_weight_path}")
    
    # 加载权重
    state_dict = torch.load(model_weight_path, map_location=DEVICE)
    if "model_state_dict" in state_dict:
        model_state_dict = state_dict["model_state_dict"]
    else:
        model_state_dict = state_dict
    
    # 处理动态创建的pos_embedding参数
    # 如果状态字典中包含pos_embedding，需要先创建它
    if "patch_embedding.pos_embedding" in model_state_dict:
        pos_embedding_tensor = model_state_dict["patch_embedding.pos_embedding"]
        
        # 检查pos_embedding的通道数是否与当前模型匹配
        saved_channels = pos_embedding_tensor.shape[1]  # (1, channels, patch_num, d_model)
        current_channels = input_channels
        
        if saved_channels != current_channels:
            print(f"⚠️ Channel mismatch: saved model has {saved_channels} channels, current model expects {current_channels}")
            print(f"   This likely means the model was trained with frequency features enabled.")
            print(f"   Enabling frequency features for inference to match training setup.")
            
            # 设置频率特征标志，确保预处理时使用频率特征
            params["freq_bands"] = True
            
            # 如果还是不匹配，需要调整pos_embedding
            if saved_channels != current_channels:
                if saved_channels > current_channels:
                    # 截取前面的通道
                    pos_embedding_tensor = pos_embedding_tensor[:, :current_channels, :, :]
                    print(f"   Truncated pos_embedding from {saved_channels} to {current_channels} channels")
                else:
                    # 重复或插值来扩展通道
                    repeat_factor = current_channels // saved_channels
                    remainder = current_channels % saved_channels
                    
                    repeated = pos_embedding_tensor.repeat(1, repeat_factor, 1, 1)
                    if remainder > 0:
                        extra = pos_embedding_tensor[:, :remainder, :, :]
                        pos_embedding_tensor = torch.cat([repeated, extra], dim=1)
                    else:
                        pos_embedding_tensor = repeated
                    print(f"   Expanded pos_embedding from {saved_channels} to {current_channels} channels")
        
        # 手动注册pos_embedding参数到模型中
        model.patch_embedding.pos_embedding = nn.Parameter(pos_embedding_tensor)
    
    # 现在可以安全地加载状态字典
    try:
        model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed: {e}")
        print("Trying non-strict loading...")
        # 如果严格加载失败，尝试非严格加载
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            # 过滤掉已知的动态参数
            filtered_unexpected = [k for k in unexpected_keys if "pos_embedding" not in k]
            if filtered_unexpected:
                print(f"⚠️ Still have unexpected keys after filtering: {filtered_unexpected}")
    
    print(f"✅ Loaded {model_type} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, params

def preprocess_eeg_data(channels: List[str], data: np.ndarray, fs: float, 
                       target_fs: int = TARGET_FS, use_freq_bands: bool = False) -> np.ndarray:
    """
    EEG数据预处理流程
    
    Args:
        channels: 通道名列表
        data: EEG数据 (channels, samples)
        fs: 采样率
        target_fs: 目标采样率
        use_freq_bands: 是否使用频率特征增强
        
    Returns:
        mdata: 预处理后的数据 (montages, samples)
    """
    # 使用6导联蒙太奇
    _, mdata, _ = get_6montages(channels, data)
    
    # 重采样到目标采样率
    if fs != target_fs:
        mdata = signal.resample_poly(mdata, target_fs, int(fs), axis=1)
    
    # 增强归一化：robust z-score
    median = np.median(mdata, axis=1, keepdims=True)
    mad = np.median(np.abs(mdata - median), axis=1, keepdims=True)
    mdata = (mdata - median) / (mad + 1e-7)
    
    # 如果启用频率特征，需要扩展到36通道以匹配训练时的模型
    if use_freq_bands:
        mdata = add_frequency_features(mdata) 
    
    return mdata

def create_sliding_windows(mdata: np.ndarray, win_samp: int, step_samp: int, 
                         use_freq_bands: bool = False) -> np.ndarray:
    """
    创建滑动窗口
    
    Args:
        mdata: 预处理后的EEG数据
        win_samp: 窗口大小（样本数）
        step_samp: 步长（样本数）
        use_freq_bands: 是否添加频率特征
        
    Returns:
        segments: 分段数据 (n_segments, channels, win_samp)
    """
    n_seg = max(0, (mdata.shape[1] - win_samp) // step_samp + 1)
    
    if n_seg == 0:
        return np.array([])
    
    segments = []
    for s in range(0, n_seg * step_samp, step_samp):
        window = mdata[:, s:s + win_samp]
        
        # 添加频率域特征（如果启用）
        if use_freq_bands:
            window = add_frequency_features(window)
        
        segments.append(window)
    
    return np.stack(segments)

def model_inference(model, segments: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """
    模型推理
    
    Args:
        model: 训练好的模型
        segments: 分段数据
        batch_size: 批处理大小
        
    Returns:
        probs: 癫痫概率数组
    """
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
            
            try:
                logits = model(batch_tensor)
                probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
            except Exception as e:
                print(f"⚠️ Inference error for batch {i//batch_size}: {e}")
                # 如果推理失败，使用默认低概率
                probs = np.zeros(len(batch))
                all_probs.extend(probs)
    
    return np.array(all_probs)

def postprocess_predictions(probs: np.ndarray, prob_th: float, min_len: int, 
                          step_sec: float, win_sec: float) -> Dict[str, Any]:
    """
    后处理预测结果
    
    Args:
        probs: 癫痫概率数组
        prob_th: 概率阈值
        min_len: 最小连续长度
        step_sec: 步长（秒）
        win_sec: 窗口长度（秒）
        
    Returns:
        result: 预测结果字典
    """
    # 平滑处理
    smooth_window = min(5, len(probs))
    if smooth_window > 1:
        smooth = np.convolve(probs, np.ones(smooth_window) / smooth_window, mode="same")
    else:
        smooth = probs
    
    # 应用阈值
    mask = smooth > prob_th
    
    # 寻找连通域
    runs, current = [], []
    for i, m in enumerate(mask):
        if m:
            current.append(i)
        elif current:
            if len(current) >= min_len:
                runs.append(current)
            current = []
    if current and len(current) >= min_len:
        runs.append(current)
    
    if not runs:  # 无癫痫检测
        return {
            "seizure_present": False,
            "seizure_confidence": float(smooth.max()) if len(smooth) > 0 else 0.0,
            "onset": -1,
            "onset_confidence": 0.0,
            "offset": -1,
            "offset_confidence": 0.0
        }
    
    # 选择置信度最高的连通域
    best_run = max(runs, key=lambda r: smooth[r].max())
    onset_sec = best_run[0] * step_sec
    offset_sec = best_run[-1] * step_sec + win_sec
    
    # 计算置信度
    conf = float(smooth[best_run].mean())
    max_conf = float(smooth[best_run].max())
    
    return {
        "seizure_present": True,
        "seizure_confidence": max_conf,
        "onset": onset_sec,
        "onset_confidence": conf,
        "offset": offset_sec,
        "offset_confidence": conf
    }

def predict_labels(channels: List[str], data: np.ndarray,
                   fs: float, reference_system: str,
                   model_name: str = "model_tf_optimized.json") -> Dict[str, Any]:
    """
    主预测函数，与主办方接口保持一致
    
    Args:
        channels: EEG通道名列表
        data: EEG数据矩阵 (channels, samples)
        fs: 采样率
        reference_system: 参考系统（"LE", "AR", "Sz"等）
        model_name: 模型元数据文件名
        
    Returns:
        result: 预测结果字典，包含癫痫检测和定位信息
    """
    try:
        # 1. 加载模型（这里会自动检测和设置频率特征标志）
        model, params = load_model_from_metadata(model_name)
        
        # 2. 获取推理参数
        prob_th = params.get("prob_th", 0.5)
        min_len = params.get("min_len", 2)
        win_sec = params.get("win_sec", WIN_SEC)
        step_sec = params.get("step_sec", STEP_SEC)
        target_fs = params.get("fs", TARGET_FS)
        use_freq_bands = params.get("freq_bands", False)  # 从模型加载时可能会被自动设置
        
        print(f"🔧 Inference settings: freq_bands={use_freq_bands}, channels={params.get('input_channels', 36)}")
        
        # 3. 数据预处理（根据模型需求决定是否使用频率特征）
        mdata = preprocess_eeg_data(channels, data, fs, target_fs, use_freq_bands)
        
        print(f"📊 Preprocessed data shape: {mdata.shape} (expected channels: {params.get('input_channels', 36)})")
        
        # 验证通道数是否匹配
        expected_channels = params.get("input_channels", 36)
        if mdata.shape[0] != expected_channels:
            print(f"⚠️ Channel count mismatch: got {mdata.shape[0]}, expected {expected_channels}")
            if mdata.shape[0] == 6 and expected_channels == 36:
                print("   Model was trained with frequency features, but inference data only has 6 montage channels")
                # 强制启用频率特征 - 修正函数调用
                mdata = add_frequency_features(mdata)  # 只传递一个参数
                print(f"   Applied frequency features, new shape: {mdata.shape}")
        
        # 4. 创建滑动窗口
        win_samp = int(win_sec * target_fs)
        step_samp = int(step_sec * target_fs)
        segments = create_sliding_windows(mdata, win_samp, step_samp, use_freq_bands=False)  # 频率特征已在预处理中添加
        
        if len(segments) == 0:  # 录音太短
            return {
                "seizure_present": False,
                "seizure_confidence": 0.0,
                "onset": -1,
                "onset_confidence": 0.0,
                "offset": -1,
                "offset_confidence": 0.0
            }
        
        print(f"📏 Created {len(segments)} segments, shape: {segments[0].shape}")
        
        # 5. 模型推理
        probs = model_inference(model, segments, batch_size=16)
        
        # 6. 后处理
        result = postprocess_predictions(probs, prob_th, min_len, step_sec, win_sec)
        
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        # 返回安全的默认结果
        return {
            "seizure_present": False,
            "seizure_confidence": 0.0,
            "onset": -1,
            "onset_confidence": 0.0,
            "offset": -1,
            "offset_confidence": 0.0
        }

# 测试函数
def test_prediction():
    """测试预测函数"""
    # 模拟数据
    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    fs = 500.0
    duration = 60  # 60秒
    n_samples = int(fs * duration)
    
    # 生成随机EEG数据
    np.random.seed(42)
    data = np.random.randn(len(channels), n_samples) * 50  # 微伏级别
    
    # 添加一些模拟的癫痫样活动
    seizure_start = int(20 * fs)
    seizure_end = int(30 * fs)
    data[:, seizure_start:seizure_end] += np.random.randn(len(channels), seizure_end - seizure_start) * 100
    
    # 进行预测
    result = predict_labels(channels, data, fs, "LE", "model_tf_optimized.json")
    
    print("🧪 测试预测结果:")
    for key, value in result.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    test_prediction()
