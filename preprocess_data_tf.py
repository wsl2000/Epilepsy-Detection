#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess EEG data for transformer training.
Saves processed data to disk to reduce memory pressure during training.
"""

import os
import numpy as np
import json
from scipy import signal
from tqdm import tqdm
from typing import List, Tuple
import multiprocessing as mp

from wettbewerb import load_references, get_6montages, EEGDataset

# Configuration
TARGET_FS = 400
WIN_SEC = 4.0
STEP_SEC = 2.0
WIN_SAMP = int(TARGET_FS * WIN_SEC)
STEP_SAMP = int(TARGET_FS * STEP_SEC)

def add_frequency_features(window):
    """Add frequency band power features."""
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
        
        # Concatenate original signal with band powers
        band_features = np.tile(band_powers[:, None], (1, window.shape[1]))
        enhanced_ch = np.vstack([ch_data[None, :], band_features])
        enhanced_window.append(enhanced_ch)
    
    return np.concatenate(enhanced_window, axis=0)

def process_single_file(args):
    """Process a single EEG file and return windows."""
    file_idx, folder, freq_bands, output_dir = args
    
    try:
        # Create dataset instance in worker
        dataset = EEGDataset(folder)
        if file_idx >= len(dataset):
            return file_idx, 0, f"Index {file_idx} out of range"
        
        # Load file data
        file_id, channels, data, fs, ref_sys, labels = dataset[file_idx]
        
        # Get montages
        _, mdata, _ = get_6montages(channels, data)
        mdata = signal.resample_poly(mdata, TARGET_FS, int(fs), axis=1)
        
        # Enhanced normalization: robust z-score
        median = np.median(mdata, axis=1, keepdims=True)
        mad = np.median(np.abs(mdata - median), axis=1, keepdims=True)
        mdata = (mdata - median) / (mad + 1e-7)

        seiz_present, onset, offset = labels
        n_seg = max(0, (mdata.shape[1] - WIN_SAMP) // STEP_SAMP + 1)
        
        if n_seg == 0:
            return file_idx, 0, f"No segments for {file_id}"
        
        # Process windows
        windows = []
        window_labels = []
        
        for k in range(n_seg):
            s = k * STEP_SAMP
            e = s + WIN_SAMP
            window = mdata[:, s:e]
            
            # Add frequency domain features if enabled
            if freq_bands:
                window = add_frequency_features(window)
            
            windows.append(window)

            if seiz_present:
                t_start = s / TARGET_FS
                t_end = e / TARGET_FS
                label = int((t_start <= offset) and (t_end >= onset))
            else:
                label = 0
            window_labels.append(label)
        
        # Save processed data
        file_output_dir = os.path.join(output_dir, file_id)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Save windows and labels as separate files
        windows_array = np.array(windows, dtype=np.float32)
        labels_array = np.array(window_labels, dtype=np.int64)
        
        np.save(os.path.join(file_output_dir, 'windows.npy'), windows_array)
        np.save(os.path.join(file_output_dir, 'labels.npy'), labels_array)
        
        # Save metadata
        metadata = {
            'file_id': file_id,
            'num_windows': len(windows),
            'window_shape': windows[0].shape,
            'seizure_present': seiz_present,
            'onset': onset,
            'offset': offset,
            'original_fs': fs,
            'target_fs': TARGET_FS,
            'ref_sys': ref_sys
        }
        
        with open(os.path.join(file_output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return file_idx, len(windows), None
        
    except Exception as e:
        return file_idx, 0, f"Error processing file {file_idx}: {str(e)}"

def preprocess_dataset(
    input_folder: str = r"D:\datasets\eeg\dataset_dir_original\shared_data\training",
    output_folder: str = r"D:\datasets\eeg\dataset_processed\wike25_tf",
    freq_bands: bool = True,
    num_workers: int = None,
    start_idx: int = 0
):
    """
    Preprocess the entire EEG dataset and save to disk.
    
    Parameters
    ----------
    input_folder : str
        Path to original EEG data
    output_folder : str
        Path to save processed data
    freq_bands : bool
        Whether to include frequency band features
    num_workers : int
        Number of worker processes
    start_idx : int
        Starting index for processing (for resuming)
    """
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get dataset size
    temp_dataset = EEGDataset(input_folder)
    total_files = len(temp_dataset)
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 4)
    
    print(f"🔄 Preprocessing {total_files - start_idx} EEG files with {num_workers} workers...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Frequency bands: {freq_bands}")
    
    # Prepare arguments for multiprocessing
    process_args = [
        (i, input_folder, freq_bands, output_folder) 
        for i in range(start_idx, total_files)
    ]
    
    total_windows = 0
    processed_files = 0
    error_count = 0
    
    # Process files
    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, process_args, chunksize=1),
                total=len(process_args),
                desc="Processing files"
            ))
    else:
        results = []
        for args in tqdm(process_args, desc="Processing files"):
            results.append(process_single_file(args))
    
    # Collect results
    for file_idx, num_windows, error in results:
        if error is not None:
            print(f"⚠️ {error}")
            error_count += 1
        else:
            total_windows += num_windows
            processed_files += 1
    
    # Save overall dataset metadata
    dataset_metadata = {
        'total_files': total_files,
        'processed_files': processed_files,
        'error_count': error_count,
        'total_windows': total_windows,
        'freq_bands': freq_bands,
        'target_fs': TARGET_FS,
        'win_sec': WIN_SEC,
        'step_sec': STEP_SEC,
        'win_samp': WIN_SAMP,
        'step_samp': STEP_SAMP,
        'input_folder': input_folder,
        'output_folder': output_folder
    }
    
    with open(os.path.join(output_folder, 'dataset_metadata.json'), 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"\n✅ Preprocessing completed!")
    print(f"📁 Processed files: {processed_files}/{total_files}")
    print(f"📊 Total windows: {total_windows:,}")
    print(f"⚠️ Errors: {error_count}")
    print(f"💾 Data saved to: {output_folder}")
    
    return dataset_metadata

if __name__ == "__main__":
    # Configure preprocessing
    config = {
        # "input_folder": r"D:\datasets\eeg\dataset_dir_original\shared_data\training",
        # "output_folder": r"D:\datasets\eeg\dataset_processed\wike25_tf",
        "input_folder": r"/work/projects/project02629/datasets/dataset_dir_original/shared_data/training",
        "output_folder": r"/work/projects/project02629/datasets/processed/wike25_tf",
        "freq_bands": True,
        "num_workers": 15,
        "start_idx": 0  # Change this to resume from a specific index
    }
    
    # Run preprocessing
    metadata = preprocess_dataset(**config)
    print(f"\n🎉 Preprocessing completed successfully!")
