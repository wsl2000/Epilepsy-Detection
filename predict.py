# -*- coding: utf-8 -*-
"""
推理脚本：载入 transformer_model.json + eeg_transformer.pt
与主办方接口保持一致。
"""

from typing import List, Dict, Any
import json, torch, torch.nn as nn, numpy as np, math
import torch.nn.functional as F
from scipy import signal
from wettbewerb import get_6montages

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- 网络结构需匹配训练 --------
class MultiHeadChannelAttention(nn.Module):
    """Channel attention mechanism for EEG."""
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction),
            nn.ReLU(),
            nn.Linear(num_channels // reduction, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = avg_out + max_out
        return x * attention.unsqueeze(-1)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal information."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EEGTransformer(nn.Module):
    """Enhanced Transformer for EEG with deep stacking capabilities like LLMs."""
    def __init__(self, input_channels=36, patch_size=16, emb_dim=256, 
                 num_heads=8, num_layers=12, dropout=0.1, model_scale="base"):
        super().__init__()
        
        # Model scale configurations (like GPT family)
        scale_configs = {
            "small": {"emb_dim": 256, "num_layers": 6, "num_heads": 8},
            "base": {"emb_dim": 512, "num_layers": 12, "num_heads": 8},
            "large": {"emb_dim": 768, "num_layers": 24, "num_heads": 12},
            "xl": {"emb_dim": 1024, "num_layers": 36, "num_heads": 16}
        }
        
        if model_scale in scale_configs:
            config = scale_configs[model_scale]
            emb_dim = config["emb_dim"]
            num_layers = config["num_layers"]
            num_heads = config["num_heads"]
        
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        
        # Calculate num_patches based on input size
        # For 4 seconds at 400 Hz = 1600 samples
        win_samp = int(4.0 * 400)  # WIN_SEC * TARGET_FS from training
        self.num_patches = win_samp // patch_size
        
        # Channel attention
        self.channel_attention = MultiHeadChannelAttention(input_channels)
        
        # Patch embedding with better initialization
        self.patch_embed = nn.Linear(input_channels * patch_size, emb_dim)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(emb_dim, self.num_patches)
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Deep Transformer encoder stack with modern improvements
        self.layers = nn.ModuleList([
            self._make_transformer_layer(emb_dim, num_heads, dropout, i) 
            for i in range(num_layers)
        ])
        
        # Layer normalization before final output (Pre-LN like modern LLMs)
        self.final_norm = nn.LayerNorm(emb_dim)
        
        # Multi-scale feature fusion
        self.conv1d_features = nn.ModuleList([
            nn.Conv1d(input_channels, emb_dim//4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]
        ])
        
        # Classification head with residual connection
        self.pre_classifier = nn.Linear(emb_dim + emb_dim, emb_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim//2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(emb_dim//2, 2)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _make_transformer_layer(self, emb_dim, num_heads, dropout, layer_idx):
        """Create a transformer layer with modern improvements."""
        # Implement scaling for deeper networks
        dropout_rate = dropout * (1.0 + 0.1 * layer_idx / self.num_layers)
        
        return nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-LN like modern transformers
        )
    
    def _init_weights(self, module):
        """Initialize weights following modern best practices."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')

    def forward(self, x, use_checkpoint=False):
        B, C, T = x.shape
        
        # Channel attention
        x = self.channel_attention(x)
        
        # Multi-scale conv features
        conv_features = []
        for conv in self.conv1d_features:
            feat = F.adaptive_avg_pool1d(conv(x), 1).squeeze(-1)
            conv_features.append(feat)
        conv_features = torch.cat(conv_features, dim=1)  # (B, emb_dim)
        
        # Patch embedding
        x = x.transpose(1, 2)  # (B, T, C)
        patches = x.unfold(1, self.patch_size, self.patch_size)  # (B, num_patches, C, patch_size)
        patches = patches.reshape(B, self.num_patches, -1)  # (B, num_patches, C*patch_size)
        
        # Embed patches
        x = self.patch_embed(patches)  # (B, num_patches, emb_dim)
        x = self.pos_encoding(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Deep transformer stack
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Global representation from class token + conv features
        cls_output = x[:, 0]  # Class token
        combined_features = torch.cat([cls_output, conv_features], dim=1)
        
        # Residual connection before classification
        pre_logits = self.pre_classifier(combined_features)
        residual = F.adaptive_avg_pool1d(conv_features.unsqueeze(-1), 1).squeeze(-1)
        if residual.shape[1] == pre_logits.shape[1]:
            pre_logits = pre_logits + residual
        
        return self.classifier(pre_logits)

def add_frequency_features(window, target_fs=400):
    """Add frequency band power features for enhanced preprocessing."""
    # EEG frequency bands: delta(0.5-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-100)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
    enhanced_window = []
    
    for ch in range(window.shape[0]):
        ch_data = window[ch]
        freqs, psd = signal.welch(ch_data, target_fs, nperseg=min(256, len(ch_data)))
        
        band_powers = []
        for low, high in bands:
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            band_powers.append(band_power)
        
        # Normalize band powers
        band_powers = np.array(band_powers)
        band_powers = band_powers / (np.sum(band_powers) + 1e-7)
        
        # Concatenate original signal with band powers (repeated to match time dimension)
        band_features = np.tile(band_powers[:, None], (1, window.shape[1]))
        enhanced_ch = np.vstack([ch_data[None, :], band_features])
        enhanced_window.append(enhanced_ch)
    
    return np.concatenate(enhanced_window, axis=0)  # (6*6, time_steps)

# -------- 接口函数 --------
def predict_labels(channels: List[str], data: np.ndarray,
                   fs: float, reference_system: str,
                   model_name: str = "transformer_model.json") -> Dict[str, Any]:

    # 1. 读取元数据 & 模型
    with open(model_name, "r") as f:
        params = json.load(f)

    # 从元数据中获取模型参数
    model_scale = params.get("model_scale", "base")
    input_channels = params.get("input_channels", 36)
    patch_size = params.get("patch_size", 16)
    emb_dim = params.get("emb_dim", 512)
    num_heads = params.get("num_heads", 8)
    num_layers = params.get("num_layers", 12)
    dropout = params.get("dropout", 0.1)

    # 创建并加载模型
    model = EEGTransformer(
        input_channels=input_channels,
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        model_scale=model_scale
    ).to(DEVICE).eval()
    
    state_dict = torch.load(params["model_weight_path"], map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    # 获取推理参数
    prob_th = params.get("prob_th", 0.5)
    min_len = params.get("min_len", 2)
    win_sec = params.get("win_sec", 4.0)
    step_sec = params.get("step_sec", 2.0)
    tgt_fs = params.get("fs", 400)
    use_freq_bands = params.get("freq_bands", True)

    # 2. 预处理 - 使用6导联蒙太奇
    _, mdata, _ = get_6montages(channels, data)
    mdata = signal.resample_poly(mdata, tgt_fs, int(fs), axis=1)
    
    # 增强归一化：robust z-score
    median = np.median(mdata, axis=1, keepdims=True)
    mad = np.median(np.abs(mdata - median), axis=1, keepdims=True)
    mdata = (mdata - median) / (mad + 1e-7)

    win_samp = int(win_sec * tgt_fs)
    step_samp = int(step_sec * tgt_fs)
    n_seg = max(0, (mdata.shape[1] - win_samp) // step_samp + 1)

    if n_seg == 0:   # 录音太短
        return {"seizure_present": False,
                "seizure_confidence": 0.,
                "onset": -1,
                "onset_confidence": 0.,
                "offset": -1,
                "offset_confidence": 0.}

    # 准备分段数据，包含频率特征
    segments = []
    for s in range(0, n_seg * step_samp, step_samp):
        window = mdata[:, s:s + win_samp]
        
        # 添加频率域特征（如果启用）
        if use_freq_bands:
            window = add_frequency_features(window, tgt_fs)
        
        segments.append(window)

    segs = np.stack(segments)
    
    # 3. 模型推理
    with torch.no_grad():
        # 分批处理以避免内存问题
        batch_size = 16
        all_probs = []
        
        for i in range(0, len(segs), batch_size):
            batch = segs[i:i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
            
            logits = model(batch_tensor)
            probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
        
        probs = np.array(all_probs)

    # 4. 平滑 + 阈值
    # 使用更长的平滑窗口来减少噪声
    smooth_window = min(5, len(probs))
    if smooth_window > 1:
        smooth = np.convolve(probs, np.ones(smooth_window) / smooth_window, mode="same")
    else:
        smooth = probs
    
    mask = smooth > prob_th

    # 连通域阈值
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

    if not runs:     # 无癫痫
        return {"seizure_present": False,
                "seizure_confidence": float(smooth.max()),
                "onset": -1, 
                "onset_confidence": 0.,
                "offset": -1, 
                "offset_confidence": 0.}

    # 取置信度最高的 run（而非最长）
    best_run = max(runs, key=lambda r: smooth[r].max())
    onset_sec = best_run[0] * step_sec
    offset_sec = best_run[-1] * step_sec + win_sec

    # 使用该区间的平均置信度
    conf = float(smooth[best_run].mean())
    max_conf = float(smooth[best_run].max())
    
    return {"seizure_present": True,
            "seizure_confidence": max_conf,
            "onset": onset_sec,
            "onset_confidence": conf,
            "offset": offset_sec,
            "offset_confidence": conf}
