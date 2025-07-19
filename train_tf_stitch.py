#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Enhanced Transformer with CBraMod-inspired tricks for EEG seizure detection.
Combines:
- CBraMod's criss-cross attention and spectral features
- Enhanced Transformer architecture
- Competition-oriented training
- Speed optimizations: mixed precision, gradient accumulation, model compilation
"""

import os, json, math, random, time
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
from scipy import signal
from tqdm import tqdm
from typing import List, Tuple, Optional, Callable, Union
import multiprocessing as mp
from einops.layers.torch import Rearrange
import copy
from train_tf import *

from wettbewerb import load_references, get_3montages, get_6montages, EEGDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FS = 400
WIN_SEC = 4.0
STEP_SEC = 2.0
WIN_SAMP = int(TARGET_FS * WIN_SEC)
STEP_SAMP = int(TARGET_FS * STEP_SEC)

# -------- 数据加载与预处理 --------
def load_single_file(args):
    """Load a single EEG file for multiprocessing."""
    idx, dataset = args
    try:
        return idx, dataset[idx], None
    except Exception as e:
        return idx, None, f"Error loading file {idx}: {str(e)}"

def load_single_file_optimized(file_idx_and_folder):
    """Load a single EEG file with minimal data transfer."""
    file_idx, folder = file_idx_and_folder
    try:
        # Create dataset instance in worker process to avoid passing large objects
        dataset = EEGDataset(folder)
        if file_idx >= len(dataset):
            return file_idx, None, f"Index {file_idx} out of range"
        
        # Load only the file data we need
        file_data = dataset[file_idx]
        file_id, file_channels, file_eeg_data, file_fs, file_ref_sys, file_labels = file_data
        
        # Return only essential data to reduce transfer size
        return file_idx, {
            'id': file_id,
            'channels': file_channels,
            'data': file_eeg_data,
            'fs': file_fs,
            'ref_sys': file_ref_sys,
            'labels': file_labels
        }, None
    except Exception as e:
        return file_idx, None, f"Error loading file {file_idx}: {str(e)}"

def load_references_modified(folder: str = '../training', idx: int = 0, num_workers: int = None) -> Tuple[List[str], List[List[str]],
                                                          List[np.ndarray],  List[float],
                                                          List[str], List[Tuple[bool,float,float]]]:
    """
    laden alle Referenzdaten aus .mat (Messdaten) und .csv (Label) Dateien ein.
    优化版本：减少内存使用并避免大数据传输问题。
    
    Parameters
    ----------
    folder : str, optional
        Ort der Trainingsdaten. Default Wert '../training'.
    idx : int, optinal
        Index, ab dem das Laden der Daten starten soll.
        z.B. idx=10 bedeutet es werden die Datenpunkte ab 10 geladen
        Falls idx >= dataset size, wird None zurückgegeben.
    num_workers : int, optional
        Number of worker threads. If None, uses CPU count.

    Returns
    -------
    ids : List[str]
        Liste von ID der Aufnahmen
    channels : List[List[str]]
        Liste der vorhandenen Kanäle per Aufnahme
    data :  List[ndarray]
        Liste der Daten pro Aufnahme
    sampling_frequencies : List[float]
        Liste der Sampling-Frequenzen.
    reference_systems : List[str]
        Liste der Referenzsysteme. "LE", "AR", "Sz" (Zusatz-Information)
    eeg_labels : List[Tuple[bool,float,float]]
        Liste der EEG Labels (seizure_present, onset, offset)
    """
    
    # Create a temporary dataset to get size
    temp_dataset = EEGDataset(folder)
    
    # Check if idx is valid
    if idx >= len(temp_dataset):
        print("Dataset is smaller than the provided idx")
        return None
    
    # Determine number of workers (reduce for memory efficiency)
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 4)  # Limit to 4 to reduce memory pressure
    
    # Load ALL data from idx to end
    total_files = len(temp_dataset) - idx
    print(f"Loading ALL {total_files} files from index {idx} to {len(temp_dataset)-1} with {num_workers} workers")
    
    # Initialisiere Listen
    ids: List[str] = []
    channels: List[List[str]] = []
    data: List[np.ndarray] = []
    sampling_frequencies: List[float] = []
    reference_systems: List[str] = []
    eeg_labels: List[Tuple[bool,float,float]] = []
    
    # Use multiprocessing with optimized data transfer
    if num_workers > 1 and total_files > 10:  # Only use multiprocessing for larger datasets
        # Prepare arguments - pass folder path instead of dataset object
        load_args = [(i, folder) for i in range(idx, len(temp_dataset))]
        
        try:
            with mp.Pool(processes=num_workers) as pool:
                # Use imap with smaller chunks to reduce memory pressure
                chunk_size = max(1, total_files // (num_workers * 4))
                results = list(tqdm(
                    pool.imap(load_single_file_optimized, load_args, chunksize=chunk_size),
                    total=len(load_args),
                    desc="Loading EEG files"
                ))
            
            # Sort results by index to maintain order
            results.sort(key=lambda x: x[0])
            
            # Process results and handle errors
            error_count = 0
            for file_idx, file_data, error in results:
                if error is not None:
                    print(f"⚠️ {error}")
                    error_count += 1
                    continue
                
                if file_data is not None:
                    ids.append(file_data['id'])
                    channels.append(file_data['channels'])
                    data.append(file_data['data'])
                    sampling_frequencies.append(file_data['fs'])
                    reference_systems.append(file_data['ref_sys'])
                    eeg_labels.append(file_data['labels'])
            
            if error_count > 0:
                print(f"⚠️ {error_count} files failed to load")
        
        except Exception as e:
            print(f"⚠️ Multiprocessing failed: {e}")
            print("Falling back to single-threaded loading...")
            num_workers = 1
    
    # Single-threaded fallback or small datasets
    if num_workers <= 1 or total_files <= 10:
        print("Using single-threaded loading...")
        dataset = EEGDataset(folder)
        for i in tqdm(range(idx, len(dataset)), desc="Loading EEG files"):
            try:
                file_data = dataset[i]
                file_id, file_channels, file_eeg_data, file_fs, file_ref_sys, file_labels = file_data
                ids.append(file_id)
                channels.append(file_channels)
                data.append(file_eeg_data)
                sampling_frequencies.append(file_fs)
                reference_systems.append(file_ref_sys)
                eeg_labels.append(file_labels)
            except Exception as e:
                print(f"⚠️ Error loading file {i}: {str(e)}")
    
    # Zeige an wie viele Daten加载 wurden
    print(f"🎉 {len(ids)} Dateien wurden erfolgreich geladen.")
    return ids, channels, data, sampling_frequencies, reference_systems, eeg_labels

def process_single_recording(args):
    """Process a single EEG recording for multiprocessing."""
    i, ids, chs, data, fs, labels, freq_bands = args
    
    try:
        # Get montages
        _, mdata, _ = get_6montages(chs[i], data[i])
        mdata = signal.resample_poly(mdata, TARGET_FS, int(fs[i]), axis=1)
        
        # Enhanced normalization: robust z-score
        median = np.median(mdata, axis=1, keepdims=True)
        mad = np.median(np.abs(mdata - median), axis=1, keepdims=True)
        mdata = (mdata - median) / (mad + 1e-7)

        seiz_present, onset, offset = labels[i]
        n_seg = max(0, (mdata.shape[1] - WIN_SAMP) // STEP_SAMP + 1)
        
        recording_windows = []
        recording_labels = []
        
        for k in range(n_seg):
            s = k * STEP_SAMP
            e = s + WIN_SAMP
            window = mdata[:, s:e]
            
            # Add frequency domain features if enabled
            if freq_bands:
                window = add_frequency_features(window)
            
            recording_windows.append(window)

            if seiz_present:
                t_start = s / TARGET_FS
                t_end = e / TARGET_FS
                label = int((t_start <= offset) and (t_end >= onset))
            else:
                label = 0
            recording_labels.append(label)
        
        return recording_windows, recording_labels, None
        
    except Exception as e:
        return None, None, f"Error processing recording {ids[i]}: {str(e)}"

def add_frequency_features(window):
    """Add frequency band power features - moved out for multiprocessing."""
    # EEG frequency bands: delta(0.5-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-100)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
    enhanced_window = []
    
    for ch in range(window.shape[0]):
        ch_data = window[ch]
        freqs, psd = signal.welch(ch_data, TARGET_FS, nperseg=min(256, len(ch_data)))
        
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

# ------------------------------------------------------------------ #
#           CBraMod-Inspired Optimized Model (High Speed)          #
# ------------------------------------------------------------------ #

class EfficientPatchEmbedding(nn.Module):
    """CBraMod-style patch embedding adapted for our data."""
    def __init__(self, input_channels=36, patch_size=16, d_model=200):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.input_channels = input_channels
        
        # CBraMod-style proj_in (adapted for our patch_size)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        
        # Spectral projection (like CBraMod but adapted for our patch_size)
        # CBraMod uses patch_size=200, we use patch_size=16, so FFT output = 16//2+1 = 9
        fft_dim = patch_size // 2 + 1
        self.spectral_proj = nn.Sequential(
            nn.Linear(fft_dim, d_model),
            nn.Dropout(0.1),
        )
        
        # Temporal feature projection to match d_model
        # proj_in outputs 25 channels, we need to project to d_model
        self.temporal_proj = nn.Linear(25, d_model)
        
        # Simplified positional encoding approach for different channel counts
        # Use learnable positional embeddings instead of complex conv operations
        self.use_learnable_pos = True
        if self.use_learnable_pos:
            # Learnable positional embedding - will be dynamically sized
            self.pos_embedding = None  # Will be created in forward pass
        else:
            # Original CBraMod-style but simplified
            self.positional_encoding = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=(3, 3), 
                         stride=(1, 1), padding=(1, 1), groups=d_model),
            )

    def _init_learnable_pos_embedding(self, batch_size, channels, patch_num, device):
        """Initialize learnable positional embedding if needed."""
        if self.pos_embedding is None or self.pos_embedding.shape[2] != patch_num:
            # Create learnable positional embedding: (1, channels, patch_num, d_model)
            self.pos_embedding = nn.Parameter(
                torch.randn(1, channels, patch_num, self.d_model, device=device) * 0.02
            ).to(device)

    def forward(self, x):
        # x shape: (B, C, T) -> reshape to CBraMod format (B, C, patch_num, patch_size)
        B, C, T = x.shape
        patch_num = T // self.patch_size
        
        # Reshape to patch format (like CBraMod)
        x_patches = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, patch_num, patch_size)
        
        # Temporal embedding through conv (like CBraMod)
        x_reshaped = x_patches.reshape(B, 1, C * patch_num, self.patch_size)
        patch_emb = self.proj_in(x_reshaped)  # (B, 25, C*patch_num, reduced_size)
        patch_emb = patch_emb.mean(dim=-1)  # (B, 25, C*patch_num)
        patch_emb = patch_emb.permute(0, 2, 1).contiguous()  # (B, C*patch_num, 25)
        
        # Project temporal features to d_model
        patch_emb = self.temporal_proj(patch_emb)  # (B, C*patch_num, d_model)
        patch_emb = patch_emb.view(B, C, patch_num, self.d_model)  # (B, C, patch_num, d_model)
        
        # Spectral embedding (like CBraMod but adapted for our patch_size)
        x_flat = x_patches.reshape(B * C * patch_num, self.patch_size)
        spectral = torch.fft.rfft(x_flat, dim=-1, norm='forward')
        spectral = torch.abs(spectral)  # (B*C*patch_num, patch_size//2+1)
        spectral_emb = self.spectral_proj(spectral)  # (B*C*patch_num, d_model)
        spectral_emb = spectral_emb.reshape(B, C, patch_num, self.d_model)
        
        # Combine temporal and spectral embeddings (like CBraMod)
        final_emb = patch_emb + spectral_emb
        
        # Positional encoding
        if self.use_learnable_pos:
            # Use learnable positional embedding
            self._init_learnable_pos_embedding(B, C, patch_num, final_emb.device)
            pos_emb = self.pos_embedding.expand(B, -1, -1, -1)  # (B, C, patch_num, d_model)
        else:
            # Use conv-based positional encoding (simplified)
            emb_for_pos = final_emb.permute(0, 3, 1, 2)  # (B, d_model, C, patch_num)
            pos_emb = self.positional_encoding(emb_for_pos)  # (B, d_model, C, patch_num)
            pos_emb = pos_emb.permute(0, 2, 3, 1)  # (B, C, patch_num, d_model)
        
        # Ensure dimensions match before addition
        if pos_emb.shape != final_emb.shape:
            print(f"Debug: pos_emb shape {pos_emb.shape} != final_emb shape {final_emb.shape}")
            # Force match dimensions using reshape/interpolation
            if pos_emb.shape[2] != final_emb.shape[2]:  # patch_num mismatch
                pos_emb = F.interpolate(
                    pos_emb.permute(0, 3, 1, 2), 
                    size=(final_emb.shape[1], final_emb.shape[2]), 
                    mode='bilinear', 
                    align_corners=False
                ).permute(0, 2, 3, 1)
        
        final_emb = final_emb + pos_emb
        
        return final_emb  # (B, C, patch_num, d_model)

class EfficientCrissCrossLayer(nn.Module):
    """Simplified criss-cross attention based on CBraMod design."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 800, dropout: float = 0.1):
        super().__init__()
        
        # CBraMod-style criss-cross attention
        self.self_attn_s = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout, batch_first=True)
        
        # Feedforward (like CBraMod)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (B, C, patch_num, d_model) - CBraMod format
        x = src
        x = x + self._criss_cross_attention(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _criss_cross_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, C, patch_num, d_model = x.shape
        
        # Split for criss-cross attention (like CBraMod)
        xs = x[:, :, :, :d_model // 2]  # Spatial part
        xt = x[:, :, :, d_model // 2:]  # Temporal part
        
        # Spatial attention (across channels)
        xs = xs.transpose(1, 2).contiguous().view(B * patch_num, C, d_model // 2)
        xs = self.self_attn_s(xs, xs, xs, need_weights=False)[0]
        xs = xs.view(B, patch_num, C, d_model // 2).transpose(1, 2)
        
        # Temporal attention (across patches)
        xt = xt.contiguous().view(B * C, patch_num, d_model // 2)
        xt = self.self_attn_t(xt, xt, xt, need_weights=False)[0]
        xt = xt.view(B, C, patch_num, d_model // 2)
        
        # Concatenate results
        x_out = torch.cat([xs, xt], dim=3)
        return self.dropout1(x_out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.dropout2(x)

class OptimizedStitchedTransformer(nn.Module):
    """Speed-optimized transformer with fewer parameters."""
    def __init__(self, input_channels=36, patch_size=16, d_model=200, 
                 num_heads=8, num_layers=6, dropout=0.1, model_scale="small"):
        super().__init__()
        
        # Even more aggressive scale configs for speed
        scale_configs = {
            "tiny": {"d_model": 128, "num_layers": 4, "num_heads": 4},
            "small": {"d_model": 200, "num_layers": 6, "num_heads": 8},
            "base": {"d_model": 256, "num_layers": 8, "num_heads": 8},
            "large": {"d_model": 512, "num_layers": 12, "num_heads": 8},
        }
        
        if model_scale in scale_configs:
            config = scale_configs[model_scale]
            d_model = config["d_model"]
            num_layers = config["num_layers"]
            num_heads = config["num_heads"]
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_channels = input_channels
        
        # Simplified patch embedding for speed
        self.patch_embedding = EfficientPatchEmbedding(input_channels, patch_size, d_model)
        
        # Simplified transformer layers
        self.transformer_layers = nn.ModuleList([
            EfficientCrissCrossLayer(
                d_model=d_model, 
                nhead=num_heads, 
                dim_feedforward=d_model * 2,  # Reduced from 4x to 2x
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Simplified output projection
        self.proj_out = nn.Linear(d_model, d_model)
        
        # Faster classifier
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b d c s'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(d_model, 2),
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x, use_checkpoint=False):
        # x shape: (B, C, T)
        
        # Patch embedding
        patch_emb = self.patch_embedding(x)  # (B, C, patch_num, d_model)
        
        # Transformer layers
        for layer in self.transformer_layers:
            if use_checkpoint and self.training:
                patch_emb = torch.utils.checkpoint.checkpoint(layer, patch_emb, use_reentrant=False)
            else:
                patch_emb = layer(patch_emb)
        
        # Output projection
        feats = self.proj_out(patch_emb)
        
        # Classification
        logits = self.classifier(feats)
        
        return logits

    def print_model_info(self):
        """Print detailed model architecture information."""
        print("🚀 Speed-Optimized Transformer:")
        print(f"   Input channels: {self.input_channels}")
        print(f"   Patch size: {self.patch_size}")
        print(f"   Model dimension: {self.d_model}")
        print(f"   Number of layers: {self.num_layers}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
        print("\n⚡ Speed Optimizations:")
        print("   - Reduced FFN dimension (2x instead of 4x)")
        print("   - Simplified patch embedding")
        print("   - Efficient criss-cross attention")
        print("   - Fast adaptive pooling classifier")

# ------------------------------------------------------------------ #
#                CBraMod-Inspired Transformer Components            #
# ------------------------------------------------------------------ #

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

class CrissCrossTransformerEncoderLayer(nn.Module):
    """CBraMod-inspired criss-cross attention layer."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True):
        super().__init__()
        
        # Criss-cross attention: split attention for spatial and temporal
        self.self_attn_spatial = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                      batch_first=batch_first)
        self.self_attn_temporal = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                       batch_first=batch_first)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (B, num_patches, d_model)
        x = src
        if self.norm_first:
            x = x + self._criss_cross_attention(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._criss_cross_attention(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _criss_cross_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Split features for criss-cross attention
        B, num_patches, d_model = x.shape
        
        # Split into two halves
        x_spatial = x[:, :, :d_model//2]  # First half for spatial attention
        x_temporal = x[:, :, d_model//2:]  # Second half for temporal attention
        
        # Spatial attention (across patches)
        x_spatial_out = self.self_attn_spatial(x_spatial, x_spatial, x_spatial, need_weights=False)[0]
        
        # Temporal attention (same as spatial in this simplified version)
        x_temporal_out = self.self_attn_temporal(x_temporal, x_temporal, x_temporal, need_weights=False)[0]
        
        # Concatenate results
        x_out = torch.cat([x_spatial_out, x_temporal_out], dim=-1)
        return self.dropout1(x_out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class SpectralPatchEmbedding(nn.Module):
    """CBraMod-inspired patch embedding with spectral features."""
    def __init__(self, input_channels=36, patch_size=16, d_model=512):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.input_channels = input_channels
        
        # Temporal embedding (similar to CBraMod's proj_in)
        self.temporal_embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        
        # Spectral embedding (inspired by CBraMod)
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size//2 + 1, d_model//2),  # FFT output size
            nn.Dropout(0.1),
        )
        
        # Final projection to d_model
        self.final_proj = nn.Linear(32 + d_model//2, d_model)
        
        # Initialize with None, will be created dynamically
        self.pos_encoding = None
        self.max_seq_len = 0

    def _init_pos_encoding(self, seq_len, device):
        """Initialize or update positional encoding based on sequence length."""
        if self.pos_encoding is None or seq_len > self.max_seq_len:
            # Create new positional encoding
            pos_encoding = torch.zeros(1, seq_len, self.d_model, device=device)
            position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * 
                               -(math.log(10000.0) / self.d_model))
            pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
            pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
            
            # Store as regular tensor, not buffer to avoid registration conflicts
            self.pos_encoding = pos_encoding
            self.max_seq_len = seq_len

    def forward(self, x):
        # x shape: (B, C, T)
        B, C, T = x.shape
        
        # Calculate number of patches
        num_patches = T // self.patch_size
        total_patches = C * num_patches
        
        # Create patches
        patches = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, num_patches, patch_size)
        
        # Temporal embedding - process patches through conv layers
        patches_reshaped = patches.reshape(B, 1, total_patches, self.patch_size)
        temporal_emb = self.temporal_embed(patches_reshaped)  # (B, 32, total_patches, reduced_patch_size)
        temporal_emb = temporal_emb.mean(dim=-1)  # (B, 32, total_patches)
        temporal_emb = temporal_emb.transpose(1, 2)  # (B, total_patches, 32)
        
        # Spectral embedding
        patches_flat = patches.reshape(B * total_patches, self.patch_size)
        spectral = torch.fft.rfft(patches_flat, dim=-1, norm='forward')
        spectral = torch.abs(spectral)  # (B*total_patches, patch_size//2+1)
        spectral_emb = self.spectral_proj(spectral)  # (B*total_patches, d_model//2)
        spectral_emb = spectral_emb.reshape(B, total_patches, -1)  # (B, total_patches, d_model//2)
        
        # Combine temporal and spectral embeddings
        combined_emb = torch.cat([temporal_emb, spectral_emb], dim=-1)  # (B, total_patches, 32+d_model//2)
        patch_emb = self.final_proj(combined_emb)  # (B, total_patches, d_model)
        
        # Initialize or update positional encoding if needed
        seq_len = patch_emb.size(1)
        self._init_pos_encoding(seq_len, patch_emb.device)
        
        # Add positional encoding
        patch_emb = patch_emb + self.pos_encoding[:, :seq_len, :]
        
        return patch_emb

class StitchedEEGTransformer(nn.Module):
    """Transformer with CBraMod-inspired features for EEG."""
    def __init__(self, input_channels=36, patch_size=16, d_model=512, 
                 num_heads=8, num_layers=12, dropout=0.1, model_scale="base"):
        super().__init__()
        
        # Model scale configurations
        scale_configs = {
            "small": {"d_model": 256, "num_layers": 6, "num_heads": 8},
            "base": {"d_model": 512, "num_layers": 12, "num_heads": 8},
            "large": {"d_model": 768, "num_layers": 24, "num_heads": 12},
            "xl": {"d_model": 1024, "num_layers": 36, "num_heads": 16}
        }
        
        if model_scale in scale_configs:
            config = scale_configs[model_scale]
            d_model = config["d_model"]
            num_layers = config["num_layers"]
            num_heads = config["num_heads"]
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_channels = input_channels
        
        # Spectral patch embedding (CBraMod-inspired)
        self.patch_embedding = SpectralPatchEmbedding(input_channels, patch_size, d_model)
        
        # Channel attention (from original transformer)
        self.channel_attention = MultiHeadChannelAttention(input_channels)
        
        # Criss-cross transformer layers
        self.transformer_layers = nn.ModuleList([
            CrissCrossTransformerEncoderLayer(
                d_model=d_model, 
                nhead=num_heads, 
                dim_feedforward=d_model * 4,
                dropout=dropout * (1.0 + 0.1 * i / num_layers),  # Scaled dropout
                norm_first=True
            )
            for i in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Multi-scale conv features (from original)
        self.conv1d_features = nn.ModuleList([
            nn.Conv1d(input_channels, d_model//4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]
        ])
        
        # Classification head with multiple strategies
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # CBraMod-style classifiers
        self.classifier_avgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 2)
        )
        
        # Combined classifier (simplified to avoid dynamic sizing issues)
        self.final_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')

    def forward(self, x, use_checkpoint=False):
        B, C, T = x.shape
        
        # Channel attention
        x_attended = self.channel_attention(x)
        
        # Multi-scale conv features
        conv_features = []
        for conv in self.conv1d_features:
            feat = F.adaptive_avg_pool1d(conv(x_attended), 1).squeeze(-1)
            conv_features.append(feat)
        conv_features = torch.cat(conv_features, dim=1)  # (B, d_model)
        
        # Spectral patch embedding
        patch_emb = self.patch_embedding(x)  # (B, num_patches, d_model)
        
        # Criss-cross transformer layers
        for layer in self.transformer_layers:
            if use_checkpoint and self.training:
                patch_emb = torch.utils.checkpoint.checkpoint(layer, patch_emb, use_reentrant=False)
            else:
                patch_emb = layer(patch_emb)
        
        patch_emb = self.final_norm(patch_emb)
        
        # Multiple classification strategies
        # 1. Average pooling (CBraMod avgpooling_patch_reps)
        cls_avgpool = self.classifier_avgpool(patch_emb.transpose(1, 2))
        
        # 2. Global representation with conv features
        global_feat = torch.cat([
            patch_emb.mean(dim=1),  # Global average
            conv_features
        ], dim=1)
        
        # Final classification
        final_logits = self.final_classifier(global_feat)
        
        # Ensemble of different approaches
        ensemble_logits = (cls_avgpool + final_logits) / 2
        
        return ensemble_logits

    def print_model_info(self):
        """Print detailed model architecture information."""
        print("🧬 CBraMod-Stitched Transformer Architecture:")
        print(f"   Input channels: {self.input_channels}")
        print(f"   Patch size: {self.patch_size}")
        print(f"   Model dimension: {self.d_model}")
        print(f"   Number of layers: {self.num_layers}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
        print("\n📊 Model Components:")
        print(f"   - Channel Attention: {sum(p.numel() for p in self.channel_attention.parameters()):,} params")
        print(f"   - Patch Embedding: {sum(p.numel() for p in self.patch_embedding.parameters()):,} params")
        print(f"   - Transformer Layers: {sum(p.numel() for p in self.transformer_layers.parameters()):,} params")
        print(f"   - Conv Features: {sum(p.numel() for p in self.conv1d_features.parameters()):,} params")
        print(f"   - Classifiers: {sum(p.numel() for p in self.classifier_avgpool.parameters()) + sum(p.numel() for p in self.final_classifier.parameters()):,} params")
        
        print(f"\n🔍 Architecture Details:")
        print(f"   - Spectral embedding uses FFT with {self.patch_size//2 + 1} frequency bins")
        print(f"   - Criss-cross attention splits {self.d_model} dimensions into {self.d_model//2} each")
        print(f"   - Multi-scale conv kernels: [3, 7, 15, 31] -> {self.d_model//4} features each")
        print(f"   - Ensemble: avgpool + global features")

# ------------------------------------------------------------------ #
#                   Enhanced Transformer Model                      #
# ------------------------------------------------------------------ #
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

# ------------------------------------------------------------------ #
#                    Competition-Oriented Loss Functions            #
# ------------------------------------------------------------------ #
class CompetitionLoss(nn.Module):
    """Competition-oriented loss function based on WKI metrics."""
    def __init__(self, alpha=0.25, gamma=2.0, interval_weight=2.0, false_positive_penalty=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.interval_weight = interval_weight
        self.false_positive_penalty = false_positive_penalty
        
    def forward(self, inputs, targets):
        # Cross entropy base
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        seizure_probs = probs[:, 1]
        
        # False positive penalty
        fp_mask = (targets == 0)
        fp_penalty = torch.mean(seizure_probs[fp_mask] ** 2) * self.false_positive_penalty
        
        # Sensitivity enhancement
        fn_mask = (targets == 1)
        sensitivity_loss = torch.mean((1 - seizure_probs[fn_mask]) ** 2)
        
        return focal_loss.mean() + fp_penalty + sensitivity_loss * self.interval_weight

# ------------------------------------------------------------------ #
#                    Enhanced Training with Tricks                  #
# ------------------------------------------------------------------ #
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup like in LLM training."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr_scale = self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def evaluate_model(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    val_loss, tp, fp, fn = 0.0, 0, 0, 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            
            logits = model(xb, use_checkpoint=False)
            loss = criterion(logits, yb)
            
            val_loss += loss.item() * xb.size(0)
            pred = logits.argmax(1)
            tp += ((pred == 1) & (yb == 1)).sum().item()
            fp += ((pred == 1) & (yb == 0)).sum().item()
            fn += ((pred == 0) & (yb == 1)).sum().item()
    
    val_sens = tp / (tp + fn + 1e-9)
    val_ppv = tp / (tp + fp + 1e-9)
    val_f1 = 2 * val_sens * val_ppv / (val_sens + val_ppv + 1e-9)
    val_loss = val_loss / len(val_loader.dataset)
    
    return val_loss, val_f1, val_sens, val_ppv

def train(params=None):
    # Default training parameters with speed optimizations
    default_params = {
        # Data parameters
        "data_folder": r"D:\datasets\eeg\dataset_processed\wike25_tf",
        "augment": True,
        "train_split": 0.9,
        "data_percentage": 0.025,  # Keep current percentage
        
        # Model parameters
        "model_scale": "small",
        "input_channels": 36,
        "patch_size": 16,
        "dropout": 0.05,
        "use_gradient_checkpointing": False,
        
        # Training parameters - speed optimized
        "epochs": 20,  # Keep current epochs
        "batch_size": 32,  # Keep current batch size
        "gradient_accumulation_steps": 4,  # Add gradient accumulation
        "val_batch_size": 64,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        
        # Loss parameters
        "loss_type": "competition",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "interval_weight": 2.0,
        "false_positive_penalty": 1.5,
        
        # Scheduler parameters
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.01,
        
        # Early stopping parameters
        "patience": 10,
        
        # Training optimization - speed focused
        "gradient_clip_norm": 0.5,
        "num_workers": 2,
        "pin_memory": True,
        "non_blocking": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        
        # Speed optimization
        "use_mixed_precision": True,
        "compile_model": True,
        
        # Frequency settings
        "eval_frequency": 1,
        "save_frequency": 2,
        "log_frequency": 50,
        
        # Random seed
        "seed": 2025,
        
        # Save parameters
        "save_model_path": "best_tf_optimized.pt",
        "save_metadata_path": "model_tf_optimized.json",
        "weights_folder": "weights/tf_optimized"
    }
    
    # Merge user params with defaults
    if params is None:
        params = default_params
    else:
        for key, value in params.items():
            default_params[key] = value
        params = default_params
    
    # Print training configuration
    print("🔧 Speed-Optimized Training Configuration:")
    for key, value in params.items():
        if key == "data_percentage" and value < 1.0:
            print(f"   {key}: {value} ⚡ (Fast training mode - using {value*100:.1f}% of data)")
        else:
            print(f"   {key}: {value}")
    print()
    
    set_seed(params["seed"])
    
    # Create weights folder
    weights_folder = params["weights_folder"]
    os.makedirs(weights_folder, exist_ok=True)
    print(f"📁 Epoch weights will be saved to: {weights_folder}")
    
    # Load datasets with optimized settings
    print("🚀 Loading preprocessed datasets...")
    train_dataset = PreprocessedEEGDataset(
        data_folder=params["data_folder"],
        augment=params["augment"],
        train_split=params["train_split"],
        is_train=True,
        seed=params["seed"],
        data_percentage=params["data_percentage"]
    )
    
    val_dataset = PreprocessedEEGDataset(
        data_folder=params["data_folder"],
        augment=False,
        train_split=params["train_split"],
        is_train=False,
        seed=params["seed"],
        data_percentage=params["data_percentage"]
    )
    
    # Create optimized data loaders
    train_dl = DataLoader(
        train_dataset, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers=params["num_workers"], 
        pin_memory=params["pin_memory"],
        persistent_workers=params["persistent_workers"],
        prefetch_factor=params["prefetch_factor"],
        drop_last=True  # For consistent batch sizes
    )
    val_dl = DataLoader(
        val_dataset, 
        batch_size=params["val_batch_size"], 
        shuffle=False, 
        num_workers=params["num_workers"], 
        pin_memory=params["pin_memory"],
        persistent_workers=params["persistent_workers"],
        prefetch_factor=params["prefetch_factor"]
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Effective batch size: {params['batch_size'] * params['gradient_accumulation_steps']}")
    
    # Create speed-optimized model
    model = OptimizedStitchedTransformer(
        input_channels=params["input_channels"],
        patch_size=params["patch_size"],
        dropout=params["dropout"],
        model_scale=params["model_scale"]
    ).to(DEVICE)
    
    # Model compilation for PyTorch 2.0+ speed boost (with better error handling)
    compilation_success = False
    if params["compile_model"] and hasattr(torch, 'compile'):
        print("🔥 Attempting to compile model for speed...")
        try:
            # Try different compilation modes in order of preference
            compilation_modes = ['default', 'reduce-overhead', 'max-autotune']
            
            for mode in compilation_modes:
                try:
                    print(f"   Trying compilation mode: {mode}")
                    compiled_model = torch.compile(model, mode=mode)
                    
                    # Test compilation with a small dummy input
                    test_input = torch.randn(1, params["input_channels"], WIN_SAMP).to(DEVICE)
                    with torch.no_grad():
                        _ = compiled_model(test_input)
                    
                    model = compiled_model
                    compilation_success = True
                    print(f"✅ Model compiled successfully with mode: {mode}")
                    break
                    
                except Exception as e:
                    print(f"   ⚠️ Compilation mode {mode} failed: {str(e)[:100]}...")
                    continue
            
            if not compilation_success:
                print("⚠️ All compilation modes failed, continuing without compilation")
                
        except Exception as e:
            print(f"⚠️ Model compilation completely failed: {e}")
            print("   Continuing without compilation...")
    else:
        print("ℹ️ Model compilation disabled or not available")
    
    # Print detailed model information
    print(f"\n" + "="*80)
    model.print_model_info()
    if compilation_success:
        print("🔥 Model is compiled for optimized execution")
    else:
        print("⚡ Model running in standard (non-compiled) mode")
    print("="*80 + "\n")
    
    # Test model with dummy input (handle both compiled and non-compiled)
    print("🔍 Testing model with dummy input...")
    dummy_input = torch.randn(2, params["input_channels"], WIN_SAMP).to(DEVICE)
    try:
        with torch.no_grad():
            dummy_output = model(dummy_input)
        print(f"✅ Model test passed! Input: {dummy_input.shape} -> Output: {dummy_output.shape}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        print("This might be due to compilation issues. Trying with a fresh model...")
        
        # Fallback: create a new model without compilation
        model = OptimizedStitchedTransformer(
            input_channels=params["input_channels"],
            patch_size=params["patch_size"],
            dropout=params["dropout"],
            model_scale=params["model_scale"]
        ).to(DEVICE)
        
        try:
            with torch.no_grad():
                dummy_output = model(dummy_input)
            print(f"✅ Fallback model test passed! Input: {dummy_input.shape} -> Output: {dummy_output.shape}")
            compilation_success = False
        except Exception as e2:
            print(f"❌ Fallback model test also failed: {e2}")
            raise e2

    # Create loss and optimizer
    criterion = CompetitionLoss(
        alpha=params["focal_alpha"], 
        gamma=params["focal_gamma"],
        interval_weight=params["interval_weight"],
        false_positive_penalty=params["false_positive_penalty"]
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
        betas=params["betas"],
        fused=True if torch.cuda.is_available() else False  # Fused optimizer for speed
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if params["use_mixed_precision"] and torch.cuda.is_available() else None
    
    # Create scheduler
    total_steps = params["epochs"] * (len(train_dl) // params["gradient_accumulation_steps"])
    warmup_steps = int(total_steps * params["warmup_ratio"])
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_steps, 
        total_steps, 
        min_lr_ratio=params["min_lr_ratio"]
    )
    
    # Training loop with speed optimizations
    best_val_f1 = 0.0
    patience_counter = 0
    
    print(f"🚀 Starting speed-optimized training for {params['epochs']} epochs...")
    print(f"⚡ Mixed precision: {params['use_mixed_precision']}")
    print(f"⚡ Gradient accumulation: {params['gradient_accumulation_steps']} steps")
    print(f"⚡ Model compiled: {compilation_success}")
    
    for ep in range(1, params["epochs"] + 1):
        # Training phase with speed monitoring
        model.train()
        train_dataset.training = True
        train_loss, tp, fp, fn = 0.0, 0, 0, 0
        
        # Speed monitoring
        epoch_start_time = time.time()
        step_times = []

        accumulation_loss = 0.0
        optimizer.zero_grad()
        
        for step, (xb, yb) in enumerate(tqdm(train_dl, desc=f"Epoch {ep} [Train]")):
            step_start_time = time.time()
            
            xb, yb = xb.to(DEVICE, non_blocking=params["non_blocking"]), yb.to(DEVICE, non_blocking=params["non_blocking"])
            
            # Mixed precision forward pass
            if params["use_mixed_precision"] and scaler is not None:
                with autocast('cuda'):
                    logits = model(xb, use_checkpoint=params["use_gradient_checkpointing"])
                    loss = criterion(logits, yb) / params["gradient_accumulation_steps"]
                
                scaler.scale(loss).backward()
                accumulation_loss += loss.item()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % params["gradient_accumulation_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["gradient_clip_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                logits = model(xb, use_checkpoint=params["use_gradient_checkpointing"])
                loss = criterion(logits, yb) / params["gradient_accumulation_steps"]
                loss.backward()
                accumulation_loss += loss.item()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % params["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["gradient_clip_norm"])
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            # Calculate metrics (use original loss scale for metrics)
            actual_loss = loss.item() * params["gradient_accumulation_steps"]
            train_loss += actual_loss * xb.size(0)
            
            with torch.no_grad():
                pred = logits.argmax(1)
                tp += ((pred == 1) & (yb == 1)).sum().item()
                fp += ((pred == 1) & (yb == 0)).sum().item()
                fn += ((pred == 0) & (yb == 1)).sum().item()
            
            # Speed monitoring
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
        # Calculate epoch metrics
        train_sens = tp / (tp + fn + 1e-9)
        train_ppv = tp / (tp + fp + 1e-9)
        train_f1 = 2 * train_sens * train_ppv / (train_sens + train_ppv + 1e-9)
        train_loss = train_loss / len(train_dataset)
        
        # Epoch speed statistics
        epoch_time = time.time() - epoch_start_time
        avg_step_time = np.mean(step_times)
        epoch_samples_per_sec = len(train_dataset) / epoch_time
        
        # Validation
        train_dataset.training = False
        val_loss, val_f1, val_sens, val_ppv = evaluate_model(model, val_dl, criterion, DEVICE)

        print(f"Epoch {ep:02d}  "
              f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Sens: {train_sens:.4f}, PPV: {train_ppv:.4f}  "
              f"Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Sens: {val_sens:.4f}, PPV: {val_ppv:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}  "
              f"Speed: {epoch_samples_per_sec:.1f} samples/sec")

        # Save epoch model (less frequently for speed)
        if ep % params["save_frequency"] == 0 or ep == params["epochs"]:
            epoch_filename = f"epoch_{ep:03d}_f1_{val_f1:.4f}_sens_{val_sens:.4f}_ppv_{val_ppv:.4f}.pt"
            epoch_model_path = os.path.join(weights_folder, epoch_filename)
            
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_f1': val_f1,
                'val_sens': val_sens,
                'val_ppv': val_ppv,
                'val_loss': val_loss,
                'train_f1': train_f1,
                'train_sens': train_sens,
                'train_ppv': train_ppv,
                'train_loss': train_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch_time': epoch_time,
                'samples_per_sec': epoch_samples_per_sec,
                'params': params
            }, epoch_model_path)
            
            print(f"💾 Epoch {ep} model saved: {epoch_filename}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_val_f1': best_val_f1,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'params': params
            }, params["save_model_path"])
            print(f"🏆 New best validation F1: {best_val_f1:.4f} - Best model updated!")
        else:
            patience_counter += 1
            print(f"⏰ Patience: {patience_counter}/{params['patience']}")
            
            if patience_counter >= params["patience"]:
                print(f"🛑 Early stopping at epoch {ep}")
                break

    # Final evaluation and save metadata
    checkpoint = torch.load(params["save_model_path"], map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_val_loss, final_val_f1, final_val_sens, final_val_ppv = evaluate_model(model, val_dl, criterion, DEVICE)
    
    # Save metadata with speed information
    metadata = {
        "model_weight_path": params["save_model_path"],
        "model_type": "OptimizedStitchedTransformer",
        "model_scale": params["model_scale"],
        "architecture": "Speed-optimized transformer with mixed precision and gradient accumulation",
        "prob_th": 0.5,
        "min_len": 2,
        "win_sec": WIN_SEC,
        "step_sec": STEP_SEC,
        "fs": TARGET_FS,
        "patch_size": params["patch_size"],
        "d_model": model.d_model,
        "num_layers": model.num_layers,
        "best_val_f1": best_val_f1,
        "final_val_f1": final_val_f1,
        "final_val_sens": final_val_sens,
        "final_val_ppv": final_val_ppv,
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "stopped_epoch": checkpoint['epoch'],
        "speed_optimizations": {
            "mixed_precision": params["use_mixed_precision"],
            "gradient_accumulation_steps": params["gradient_accumulation_steps"],
            "model_compiled": params["compile_model"],
            "effective_batch_size": params["batch_size"] * params["gradient_accumulation_steps"]
        },
        "training_params": params
    }
    
    with open(params["save_metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎯 Speed-Optimized Training Complete!")
    print(f"🏆 Best validation F1: {best_val_f1:.4f}")
    print(f"📊 Final metrics - F1: {final_val_f1:.4f}, Sens: {final_val_sens:.4f}, PPV: {final_val_ppv:.4f}")
    print(f"⚡ Speed optimizations applied: Mixed precision, gradient accumulation, model compilation")
    print(f"📁 Model saved: {params['save_model_path']}")
    
    return {
        "best_val_f1": best_val_f1,
        "final_metrics": {
            "f1": final_val_f1,
            "sensitivity": final_val_sens,
            "ppv": final_val_ppv,
            "loss": final_val_loss
        },
        "model_path": params["save_model_path"],
        "metadata_path": params["save_metadata_path"]
    }

if __name__ == "__main__":
    # Check preprocessed data
    preprocessed_folder = r"D:\datasets\eeg\dataset_processed\wike25_tf"
    if not os.path.exists(os.path.join(preprocessed_folder, 'dataset_metadata.json')):
        print("❌ Preprocessed data not found!")
        print(f"Please run preprocess_data.py first to create data in: {preprocessed_folder}")
        exit(1)
    
    # Speed-optimized training parameters (disable compilation by default)
    custom_params = {
        # Data parameters (keep current)
        "data_folder": preprocessed_folder,
        "augment": True,
        "train_split": 0.9,
        "data_percentage": 0.025,  # Keep current 2.5%
        
        # Model parameters (optimize for speed)
        "model_scale": "small",  # Keep small for speed
        "input_channels": 36,
        "patch_size": 16,
        "dropout": 0.05,
        "use_gradient_checkpointing": False,  # Disable for speed
        
        # Training parameters (keep current epochs and batch size)
        "epochs": 20,  # Keep current
        "batch_size": 32,  # Keep current
        "gradient_accumulation_steps": 4,  # 4x effective batch size
        "val_batch_size": 64,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        
        # Loss parameters
        "loss_type": "competition",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "interval_weight": 2.0,
        "false_positive_penalty": 1.5,
        
        # Scheduler parameters
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.01,
        
        # Early stopping parameters
        "patience": 10,
        
        # Training optimization
        "gradient_clip_norm": 0.5,
        "num_workers": 2,
        "pin_memory": True,
        "non_blocking": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        
        # Speed optimization (disable problematic compilation)
        "use_mixed_precision": True,
        "compile_model": False,  # Disabled due to Triton dependency issues
        
        # Frequency settings
        "eval_frequency": 1,
        "save_frequency": 2,
        "log_frequency": 50,
        
        # Random seed
        "seed": 2025,
        
        # Save parameters
        "save_model_path": "best_tf_optimized.pt",
        "save_metadata_path": "model_tf_optimized.json",
        "weights_folder": "weights/tf_optimized"
    }
    
    print(f"🚀 Starting SPEED-OPTIMIZED training:")
    print(f"   Epochs: {custom_params['epochs']}")
    print(f"   Data: {custom_params['data_percentage']*100:.1f}% of dataset")
    print(f"   Batch size: {custom_params['batch_size']} x {custom_params['gradient_accumulation_steps']} accumulation = {custom_params['batch_size'] * custom_params['gradient_accumulation_steps']} effective")
    print(f"   Mixed precision: {custom_params['use_mixed_precision']}")
    print(f"   Model compilation: {custom_params['compile_model']} (disabled due to Triton dependency)")
    print(f"   Note: You can enable compilation by setting compile_model=True if Triton is available")
    
    results = train(custom_params)
    
    print(f"\n🏁 Training completed! Best F1: {results['best_val_f1']:.4f}")
    print(f"📁 Best model saved: {results['model_path']}")
    print(f"📄 Metadata saved: {results['metadata_path']}")

