#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Enhanced Transformer for EEG seizure detection with domain-specific tricks.
Outputs:
  ├── eeg_transformer.pt  # 网络权重
  └── transformer_model.json  # 元数据
"""

import os, json, math, random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal
from tqdm import tqdm
from typing import List, Tuple
import concurrent.futures
from functools import partial
import multiprocessing as mp

from wettbewerb import load_references, get_3montages, get_6montages, EEGDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FS = 400
WIN_SEC = 4.0
STEP_SEC = 2.0
WIN_SAMP = int(TARGET_FS * WIN_SEC)
STEP_SAMP = int(TARGET_FS * STEP_SEC)

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
#                    Preprocessed EEG Dataset                       #
# ------------------------------------------------------------------ #
class PreprocessedEEGDataset(Dataset):
    """Dataset for loading preprocessed EEG data from disk."""
    def __init__(self, data_folder: str = r"D:\datasets\eeg\dataset_processed\wike25_tf", 
                 augment: bool = True, train_split: float = 0.9, is_train: bool = True, 
                 seed: int = 2025, data_percentage: float = 1.0):
        
        self.data_folder = data_folder
        self.augment = augment
        self.is_train = is_train
        self.training = is_train  # Add training attribute for compatibility
        
        # Load dataset metadata
        metadata_path = os.path.join(data_folder, 'dataset_metadata.json')
        with open(metadata_path, 'r') as f:
            self.dataset_metadata = json.load(f)
        
        print(f"📂 Loading preprocessed data from: {data_folder}")
        print(f"📊 Full dataset info: {self.dataset_metadata['processed_files']} files, {self.dataset_metadata['total_windows']:,} windows")
        
        # Find all processed files
        self.file_paths = []
        self.file_metadata = []
        
        for item in os.listdir(data_folder):
            item_path = os.path.join(data_folder, item)
            if os.path.isdir(item_path) and item != '__pycache__':
                metadata_file = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    self.file_paths.append(item_path)
                    self.file_metadata.append(metadata)
        
        # Apply data percentage limitation BEFORE train/val split
        if data_percentage < 1.0:
            np.random.seed(seed)
            total_files = len(self.file_paths)
            subset_size = int(total_files * data_percentage)
            subset_indices = np.random.choice(total_files, subset_size, replace=False)
            
            self.file_paths = [self.file_paths[i] for i in subset_indices]
            self.file_metadata = [self.file_metadata[i] for i in subset_indices]
            
            print(f"🎯 Using {data_percentage*100:.1f}% of data: {len(self.file_paths)} files (from {total_files} total)")
        
        # Split files into train/val
        np.random.seed(seed)
        total_files = len(self.file_paths)
        indices = np.random.permutation(total_files)
        
        train_size = int(train_split * total_files)
        if is_train:
            selected_indices = indices[:train_size]
        else:
            selected_indices = indices[train_size:]
        
        self.file_paths = [self.file_paths[i] for i in selected_indices]
        self.file_metadata = [self.file_metadata[i] for i in selected_indices]
        
        # Build index mapping (file_idx, window_idx)
        self.index_mapping = []
        total_windows = 0
        
        for file_idx, metadata in enumerate(self.file_metadata):
            num_windows = metadata['num_windows']
            for window_idx in range(num_windows):
                self.index_mapping.append((file_idx, window_idx))
            total_windows += num_windows
        
        print(f"🎯 Split: {'Train' if is_train else 'Val'}")
        print(f"📁 Files: {len(self.file_paths)}")
        print(f"📊 Windows: {total_windows:,}")
        
        if data_percentage < 1.0:
            print(f"⚡ Fast training mode: Using {data_percentage*100:.1f}% of available data")
        
        # Calculate class distribution
        self._calculate_class_distribution()
    
    def _calculate_class_distribution(self):
        """Calculate and print class distribution."""
        seizure_windows = 0
        total_windows = 0
        
        for file_idx in range(len(self.file_paths)):
            labels_path = os.path.join(self.file_paths[file_idx], 'labels.npy')
            labels = np.load(labels_path)
            seizure_windows += np.sum(labels)
            total_windows += len(labels)
        
        normal_windows = total_windows - seizure_windows
        print(f"📊 Class distribution:")
        print(f"   Seizure windows: {seizure_windows:,} ({seizure_windows/total_windows*100:.2f}%)")
        print(f"   Normal windows: {normal_windows:,} ({normal_windows/total_windows*100:.2f}%)")
        if seizure_windows > 0:
            print(f"   Class ratio (seizure:normal) = 1:{normal_windows/seizure_windows:.1f}")
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        file_idx, window_idx = self.index_mapping[idx]
        
        # Load window data
        windows_path = os.path.join(self.file_paths[file_idx], 'windows.npy')
        labels_path = os.path.join(self.file_paths[file_idx], 'labels.npy')
        
        # Load specific window (memory efficient)
        windows = np.load(windows_path, mmap_mode='r')
        labels = np.load(labels_path, mmap_mode='r')
        
        x = torch.tensor(windows[window_idx], dtype=torch.float32)
        y = torch.tensor(labels[window_idx], dtype=torch.long)
        
        # Data augmentation during training
        if self.augment and self.training:  # Use self.training instead of self.is_train
            x = self._augment_data(x)
        
        return x, y
    
    def _augment_data(self, x):
        """EEG-specific data augmentation."""
        if random.random() < 0.3:  # Time masking
            mask_len = random.randint(10, 50)
            start = random.randint(0, x.shape[1] - mask_len)
            x[:, start:start+mask_len] *= 0.1
        
        if random.random() < 0.3:  # Gaussian noise
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        
        if random.random() < 0.2:  # Time shift
            shift = random.randint(-20, 20)
            x = torch.roll(x, shift, dims=1)
        
        return x

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
        self.num_patches = WIN_SAMP // patch_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        
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
        
        # Deep transformer stack with optional gradient checkpointing
        for i, layer in enumerate(self.layers):
            if use_checkpoint and self.training and i > 0:
                # Use gradient checkpointing for memory efficiency
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
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

# ------------------------------------------------------------------ #
#                    Competition-Oriented Loss Functions            #
# ------------------------------------------------------------------ #
class CompetitionLoss(nn.Module):
    """Competition-oriented loss function based on WKI metrics."""
    def __init__(self, alpha=0.25, gamma=2.0, interval_weight=2.0, false_positive_penalty=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.interval_weight = interval_weight  # Weight for interval-based accuracy
        self.false_positive_penalty = false_positive_penalty  # Penalty for false positives
        
        # Standard focal loss for comparison
        self.focal_loss = FocalLoss(alpha, gamma)
        
    def forward(self, inputs, targets):
        # Standard focal loss component
        focal_component = self.focal_loss(inputs, targets)
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        seizure_probs = probs[:, 1]  # Probability of seizure class
        
        # False positive penalty: penalize high confidence predictions on normal samples
        fp_mask = (targets == 0)  # Normal samples
        fp_penalty = torch.mean(seizure_probs[fp_mask] ** 2) * self.false_positive_penalty
        
        # Sensitivity enhancement: ensure we don't miss seizures
        fn_mask = (targets == 1)  # Seizure samples
        sensitivity_loss = torch.mean((1 - seizure_probs[fn_mask]) ** 2)
        
        # Combined loss
        total_loss = focal_component + fp_penalty + sensitivity_loss * self.interval_weight
        
        return total_loss

class IntervalAwareLoss(nn.Module):
    """Loss function that considers temporal intervals for seizure detection."""
    def __init__(self, base_loss_weight=1.0, confidence_weight=0.5, balance_weight=2.0):
        super().__init__()
        self.base_loss_weight = base_loss_weight
        self.confidence_weight = confidence_weight
        self.balance_weight = balance_weight
        
    def forward(self, inputs, targets):
        # Cross entropy base loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get prediction probabilities
        probs = F.softmax(inputs, dim=1)
        seizure_probs = probs[:, 1]
        
        # Class balancing based on competition metrics
        # Seizure samples get higher weight (sensitivity is crucial)
        seizure_mask = (targets == 1)
        normal_mask = (targets == 0)
        
        # Weight seizures more heavily to improve sensitivity
        weighted_loss = torch.zeros_like(ce_loss)
        if seizure_mask.any():
            weighted_loss[seizure_mask] = ce_loss[seizure_mask] * self.balance_weight
        if normal_mask.any():
            weighted_loss[normal_mask] = ce_loss[normal_mask]
        
        # Confidence regularization to avoid overconfident false positives
        confidence_penalty = torch.mean(torch.max(probs, dim=1)[0] ** 2) * self.confidence_weight
        
        return torch.mean(weighted_loss) * self.base_loss_weight + confidence_penalty

def evaluate_model_competition(model, val_loader, criterion, device):
    """Evaluate model with competition-specific metrics."""
    model.eval()
    val_loss, tp, fp, fn, tn = 0.0, 0, 0, 0, 0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            
            logits = model(xb, use_checkpoint=False)
            loss = criterion(logits, yb)
            
            val_loss += loss.item() * xb.size(0)
            
            # Get probabilities for more detailed analysis
            probs = F.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Seizure probabilities
            all_targets.extend(yb.cpu().numpy())
            
            pred = logits.argmax(1)
            tp += ((pred == 1) & (yb == 1)).sum().item()
            fp += ((pred == 1) & (yb == 0)).sum().item()
            fn += ((pred == 0) & (yb == 1)).sum().item()
            tn += ((pred == 0) & (yb == 0)).sum().item()
    
    # Standard metrics
    val_sens = tp / (tp + fn + 1e-9)
    val_ppv = tp / (tp + fp + 1e-9)
    val_f1 = 2 * val_sens * val_ppv / (val_sens + val_ppv + 1e-9)
    val_loss = val_loss / len(val_loader.dataset)
    
    # Competition-specific metrics
    # Simulate interval-based accuracy (approximate)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Use different thresholds to optimize for competition metrics
    best_competition_score = 0.0
    best_threshold = 0.5
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        pred_binary = (all_probs >= threshold).astype(int)
        
        tp_t = np.sum((pred_binary == 1) & (all_targets == 1))
        fp_t = np.sum((pred_binary == 1) & (all_targets == 0))
        fn_t = np.sum((pred_binary == 0) & (all_targets == 1))
        tn_t = np.sum((pred_binary == 0) & (all_targets == 0))
        
        if tp_t + fn_t > 0 and tp_t + fp_t > 0:
            sens_t = tp_t / (tp_t + fn_t)
            ppv_t = tp_t / (tp_t + fp_t)
            f1_t = 2 * sens_t * ppv_t / (sens_t + ppv_t) if (sens_t + ppv_t) > 0 else 0
            
            # Simulate competition score (higher F1 with bias toward sensitivity)
            competition_score = f1_t * (1 + 0.2 * sens_t)  # Slight bias toward sensitivity
            
            if competition_score > best_competition_score:
                best_competition_score = competition_score
                best_threshold = threshold
    
    return val_loss, val_f1, val_sens, val_ppv, best_competition_score, best_threshold

# ------------------------------------------------------------------ #
#                    Enhanced Training with Tricks                  #
# ------------------------------------------------------------------ #
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

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
            # Warmup phase
            lr_scale = self.step_count / self.warmup_steps
        else:
            # Cosine annealing phase
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
            
            logits = model(xb, use_checkpoint=False)  # No checkpointing during eval
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
    # Default training parameters
    default_params = {
        # Data parameters
        "data_folder": r"D:\datasets\eeg\dataset_processed\wike25_tf",
        "augment": True,
        "train_split": 0.9,
        "data_percentage": 1.0,  # New parameter: percentage of data to use (0.1 = 10%, 1.0 = 100%)
        
        # Model parameters
        "model_scale": "base",  # "small", "base", "large", "xl"
        "input_channels": 36,
        "patch_size": 16,
        "dropout": 0.1,
        "use_gradient_checkpointing": True,
        
        # Training parameters
        "epochs": 20,
        "batch_size": 64,  # Can increase since data loading is more efficient
        "val_batch_size": 128,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        
        # Loss parameters - Updated for competition
        "loss_type": "competition",  # "focal", "competition", "interval_aware"
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "interval_weight": 2.0,
        "false_positive_penalty": 1.5,
        "balance_weight": 2.0,
        
        # Scheduler parameters
        "warmup_ratio": 0.1,
        "min_lr_ratio": 0.05,
        
        # Early stopping parameters
        "patience": 15,
        "early_stopping_metric": "competition_score",  # "val_f1" or "competition_score"
        
        # Training optimization
        "gradient_clip_norm": 1.0,
        "num_workers": 4,  # Can increase since no heavy preprocessing
        "pin_memory": True,
        
        # Random seed
        "seed": 2025,
        
        # Save parameters
        "save_model_path": "eeg_transformer.pt",
        "save_metadata_path": "transformer_model.json",
        "weights_folder": "weights/tf"  # New parameter for epoch weights
    }
    
    # Merge user params with defaults
    if params is None:
        params = default_params
    else:
        for key, value in params.items():
            default_params[key] = value
        params = default_params
    
    # Print training configuration with data percentage highlight
    print("🔧 Training Configuration:")
    for key, value in params.items():
        if key == "data_percentage" and value < 1.0:
            print(f"   {key}: {value} ⚡ (Fast training mode - using {value*100:.1f}% of data)")
        else:
            print(f"   {key}: {value}")
    print()
    
    set_seed(params["seed"])
    
    # Create weights folder for epoch checkpoints
    weights_folder = params["weights_folder"]
    os.makedirs(weights_folder, exist_ok=True)
    print(f"📁 Epoch weights will be saved to: {weights_folder}")
    
    # Load preprocessed datasets with data percentage control
    print("🚀 Loading preprocessed datasets...")
    train_dataset = PreprocessedEEGDataset(
        data_folder=params["data_folder"],
        augment=params["augment"],
        train_split=params["train_split"],
        is_train=True,
        seed=params["seed"],
        data_percentage=params["data_percentage"]  # Add data percentage parameter
    )
    
    val_dataset = PreprocessedEEGDataset(
        data_folder=params["data_folder"],
        augment=False,  # No augmentation for validation
        train_split=params["train_split"],
        is_train=False,
        seed=params["seed"],
        data_percentage=params["data_percentage"]  # Add data percentage parameter
    )
    
    # Create data loaders
    train_dl = DataLoader(
        train_dataset, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers=params["num_workers"], 
        pin_memory=params["pin_memory"]
    )
    val_dl = DataLoader(
        val_dataset, 
        batch_size=params["val_batch_size"], 
        shuffle=False, 
        num_workers=params["num_workers"], 
        pin_memory=params["pin_memory"]
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Create model
    model = EEGTransformer(
        input_channels=params["input_channels"],
        patch_size=params["patch_size"],
        dropout=params["dropout"],
        model_scale=params["model_scale"]
    ).to(DEVICE)
    
    print(f"Model scale: {params['model_scale']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss function and optimizer
    if params["loss_type"] == "competition":
        criterion = CompetitionLoss(
            alpha=params["focal_alpha"], 
            gamma=params["focal_gamma"],
            interval_weight=params["interval_weight"],
            false_positive_penalty=params["false_positive_penalty"]
        )
        print(f"📊 Using Competition Loss (interval_weight={params['interval_weight']}, fp_penalty={params['false_positive_penalty']})")
    elif params["loss_type"] == "interval_aware":
        criterion = IntervalAwareLoss(
            balance_weight=params["balance_weight"]
        )
        print(f"📊 Using Interval-Aware Loss (balance_weight={params['balance_weight']})")
    else:
        criterion = FocalLoss(alpha=params["focal_alpha"], gamma=params["focal_gamma"])
        print(f"📊 Using Focal Loss (alpha={params['focal_alpha']}, gamma={params['focal_gamma']})")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
        betas=params["betas"]
    )
    
    # Create learning rate scheduler
    total_steps = params["epochs"] * len(train_dl)
    warmup_steps = int(total_steps * params["warmup_ratio"])
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_steps, 
        total_steps, 
        min_lr_ratio=params["min_lr_ratio"]
    )
    
    # Early stopping setup with competition-aware metrics
    best_val_score = 0.0
    best_threshold = 0.5
    patience_counter = 0
    
    print(f"🚀 Starting training for {params['epochs']} epochs...")
    print(f"🎯 Early stopping based on: {params['early_stopping_metric']}")
    
    for ep in range(1, params["epochs"] + 1):
        # Training phase
        model.train()
        train_dataset.training = True  # Set training flag directly on dataset
        train_loss, tp, fp, fn = 0.0, 0, 0, 0

        for step, (xb, yb) in enumerate(tqdm(train_dl, desc=f"Epoch {ep} [Train]")):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits = model(xb, use_checkpoint=params["use_gradient_checkpointing"])
            loss = criterion(logits, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["gradient_clip_norm"])
            
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * xb.size(0)
            pred = logits.argmax(1)
            tp += ((pred == 1) & (yb == 1)).sum().item()
            fp += ((pred == 1) & (yb == 0)).sum().item()
            fn += ((pred == 0) & (yb == 1)).sum().item()

        # Calculate training metrics
        train_sens = tp / (tp + fn + 1e-9)
        train_ppv = tp / (tp + fp + 1e-9)
        train_f1 = 2 * train_sens * train_ppv / (train_sens + train_ppv + 1e-9)
        train_loss = train_loss / len(train_dataset)
        
        # Validation phase with competition metrics
        train_dataset.training = False  # Set training flag to False for validation
        val_loss, val_f1, val_sens, val_ppv, competition_score, optimal_threshold = evaluate_model_competition(
            model, val_dl, criterion, DEVICE
        )

        print(f"Epoch {ep:02d}  "
              f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Sens: {train_sens:.4f}, PPV: {train_ppv:.4f}  "
              f"Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Sens: {val_sens:.4f}, PPV: {val_ppv:.4f}  "
              f"CompScore: {competition_score:.4f}, OptThresh: {optimal_threshold:.3f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save model for current epoch with detailed filename
        epoch_filename = f"epoch_{ep:03d}_f1_{val_f1:.4f}_comp_{competition_score:.4f}_sens_{val_sens:.4f}_ppv_{val_ppv:.4f}.pt"
        epoch_model_path = os.path.join(weights_folder, epoch_filename)
        
        # Save current epoch model
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_sens': val_sens,
            'val_ppv': val_ppv,
            'val_loss': val_loss,
            'competition_score': competition_score,
            'optimal_threshold': optimal_threshold,
            'train_f1': train_f1,
            'train_sens': train_sens,
            'train_ppv': train_ppv,
            'train_loss': train_loss,
            'learning_rate': scheduler.get_last_lr()[0],
            'params': params
        }, epoch_model_path)
        
        print(f"💾 Epoch {ep} model saved: {epoch_filename}")

        # Early stopping based on selected metric
        current_score = competition_score if params["early_stopping_metric"] == "competition_score" else val_f1
        
        if current_score > best_val_score:
            best_val_score = current_score
            best_threshold = optimal_threshold
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': val_f1,
                'best_competition_score': competition_score,
                'best_threshold': optimal_threshold,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'params': params
            }, params["save_model_path"])
            metric_name = "Competition Score" if params["early_stopping_metric"] == "competition_score" else "F1"
            print(f"🏆 New best {metric_name}: {best_val_score:.4f} - Best model updated!")
        else:
            patience_counter += 1
            print(f"⏰ Patience: {patience_counter}/{params['patience']}")
            
            if patience_counter >= params["patience"]:
                print(f"🛑 Early stopping at epoch {ep}")
                print(f"🏆 Best {params['early_stopping_metric']}: {best_val_score:.4f}")
                break

    # Load best model for final evaluation
    checkpoint = torch.load(params["save_model_path"], map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation evaluation
    final_val_loss, final_val_f1, final_val_sens, final_val_ppv = evaluate_model(model, val_dl, criterion, DEVICE)
    
    # Save model metadata
    metadata = {
        "model_weight_path": params["save_model_path"],
        "model_type": "EEGTransformer",
        "model_scale": params["model_scale"],
        "prob_th": best_threshold,  # Use optimal threshold
        "min_len": 2,
        "win_sec": WIN_SEC,
        "step_sec": STEP_SEC,
        "fs": TARGET_FS,
        "patch_size": params["patch_size"],
        "emb_dim": model.emb_dim,
        "num_heads": 8,
        "num_layers": model.num_layers,
        "best_val_f1": checkpoint.get('best_val_f1', val_f1),
        "best_competition_score": checkpoint.get('best_competition_score', 0.0),
        "optimal_threshold": checkpoint.get('best_threshold', 0.5),
        "final_val_f1": final_val_f1,
        "final_val_sens": final_val_sens,
        "final_val_ppv": final_val_ppv,
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "stopped_epoch": checkpoint['epoch'],
        "training_params": params,
        "epoch_checkpoints": {
            "folder": weights_folder,
            "total_epochs_saved": len(epoch_files),
            "checkpoint_files": epoch_files
        }
    }
    
    with open(params["save_metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Add training summary with epoch checkpoints info
    epoch_files = [f for f in os.listdir(weights_folder) if f.startswith('epoch_') and f.endswith('.pt')]
    epoch_files.sort()
    
    print(f"\n📚 Training Summary:")
    print(f"🏆 Best validation F1: {best_val_score:.4f}")
    print(f"📁 Epoch checkpoints saved: {len(epoch_files)} files in {weights_folder}")
    print(f"📊 最终验证指标:")
    print(f"   - F1: {final_val_f1:.4f}")
    print(f"   - Sensitivity: {final_val_sens:.4f}")
    print(f"   - PPV: {final_val_ppv:.4f}")
    print(f"   - Loss: {final_val_loss:.4f}")
    print(f"📁 模型已保存: {params['save_model_path']} / {params['save_metadata_path']}")
    print(f"🔢 模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Update metadata to include epoch checkpoints info
    metadata = {
        "model_weight_path": params["save_model_path"],
        "model_type": "EEGTransformer",
        "model_scale": params["model_scale"],
        "prob_th": best_threshold,  # Use optimal threshold
        "min_len": 2,
        "win_sec": WIN_SEC,
        "step_sec": STEP_SEC,
        "fs": TARGET_FS,
        "patch_size": params["patch_size"],
        "emb_dim": model.emb_dim,
        "num_heads": 8,
        "num_layers": model.num_layers,
        "best_val_f1": checkpoint.get('best_val_f1', val_f1),
        "best_competition_score": checkpoint.get('best_competition_score', 0.0),
        "optimal_threshold": checkpoint.get('best_threshold', 0.5),
        "final_val_f1": final_val_f1,
        "final_val_sens": final_val_sens,
        "final_val_ppv": final_val_ppv,
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "stopped_epoch": checkpoint['epoch'],
        "training_params": params,
        "epoch_checkpoints": {
            "folder": weights_folder,
            "total_epochs_saved": len(epoch_files),
            "checkpoint_files": epoch_files
        }
    }
    
    with open(params["save_metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "best_val_f1": best_val_score,
        "final_metrics": {
            "f1": final_val_f1,
            "sensitivity": final_val_sens,
            "ppv": final_val_ppv,
            "loss": final_val_loss
        },
        "model_path": params["save_model_path"],
        "metadata_path": params["save_metadata_path"],
        "epoch_checkpoints": {
            "folder": weights_folder,
            "files": epoch_files
        }
    }

if __name__ == "__main__":
    # First, check if preprocessed data exists
    preprocessed_folder = r"D:\datasets\eeg\dataset_processed\wike25_tf"
    if not os.path.exists(os.path.join(preprocessed_folder, 'dataset_metadata.json')):
        print("❌ Preprocessed data not found!")
        print(f"Please run preprocess_data.py first to create data in: {preprocessed_folder}")
        exit(1)
    
    # Training with competition-optimized parameters
    custom_params = {
        "epochs": 10,
        "model_scale": "small",
        "batch_size": 128,
        "learning_rate": 3e-4,
        "weights_folder": "weights/tf",
        "data_percentage": 1,
        "patience": 5,
        # Competition-specific parameters
        "loss_type": "competition",  # Use competition-aware loss
        "early_stopping_metric": "competition_score",  # Stop based on competition score
        "interval_weight": 2.0,  # Higher weight for interval accuracy
        "false_positive_penalty": 1.5,  # Penalize false positives
        "balance_weight": 2.0  # Balance classes for better sensitivity
    }
    
    print("⚡ Competition-Optimized Training Mode!")
    print(f"📊 Loss: {custom_params['loss_type']}")
    print(f"🎯 Early stopping: {custom_params['early_stopping_metric']}")
    print(f"📊 Using {custom_params['data_percentage']*100:.1f}% of available data")
    
    results = train(custom_params)
    print(f"\n🎉 Training completed! Best score: {results.get('best_competition_score', 'N/A')}")
    print(f"📁 Epoch checkpoints available in: {results['epoch_checkpoints']['folder']}")              