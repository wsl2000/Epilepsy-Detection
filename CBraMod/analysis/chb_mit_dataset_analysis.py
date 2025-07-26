import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyedflib
import pyedflib.highlevel as hl
from collections import defaultdict, Counter
from scipy import signal
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# 设置全局字体为 SimHei (黑体) 或其他中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

class CHBMITDatasetAnalyzer:
    def __init__(self, 
                 original_path="/data/datasets/chb-mit-scalp-eeg-database-1.0.0",
                 processed_path="/data/datasets/BigDownstream/chb-mit/processed",
                 segmented_path="/data/datasets/BigDownstream/chb-mit/processed_seg",
                 output_dir="./analysis_results"):
        self.original_path = original_path
        self.processed_path = processed_path
        self.segmented_path = segmented_path
        self.output_dir = output_dir
        
        # Parameters from preprocessing scripts
        self.channels = [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
        ]
        self.SAMPLING_RATE = 256
        self.test_pats = ["chb23", "chb24"]
        self.val_pats = ["chb21", "chb22"]
        self.train_pats = [
            "chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
            "chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb19", "chb20"
        ]
        
        # Processing parameters from process1.py
        self.process1_params = [
            ("01", "01", 2, 46, 0), ("02", "01", 2, 35, 0), ("03", "01", 2, 38, 0), ("05", "01", 2, 39, 0),
            ("06", "01", 2, 24, 0), ("07", "01", 2, 19, 0), ("08", "02", 3, 29, 0), ("10", "01", 2, 89, 0),
            ("11", "01", 2, 99, 0), ("14", "01", 2, 42, 0), ("20", "01", 2, 68, 0), ("21", "01", 2, 33, 0),
            ("22", "01", 2, 77, 0), ("23", "06", 7, 20, 0), ("24", "01", 3, 21, 0), ("04", "07", 1, 43, 1),
            ("09", "02", 1, 19, 1), ("15", "02", 1, 63, 1), ("16", "01", 2, 19, 0), ("18", "02", 1, 36, 1),
            ("19", "02", 1, 30, 1)
        ]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def analyze_edf_file(self, edf_path):
        """Analyze a single EDF file"""
        try:
            signals, signal_headers, header = hl.read_edf(edf_path, digital=False)
            
            file_stats = {
                'file_size_mb': os.path.getsize(edf_path) / (1024 * 1024),
                'duration': header.get('Duration', 0) if isinstance(header, dict) else getattr(header, 'duration', 0),
                'channel_count': len(signal_headers),
                'sampling_rates': [],
                'channel_names': [],
                'signal_stats': []
            }
            
            # Analyze each channel
            for i, (sig, h) in enumerate(zip(signals, signal_headers)):
                if isinstance(h, dict):
                    sampling_rate = h.get('sample_rate') or h.get('sample_frequency') or self.SAMPLING_RATE
                    channel_name = h.get('label', f'Channel_{i}')
                else:
                    sampling_rate = getattr(h, 'sample_rate', None) or getattr(h, 'sample_frequency', None) or self.SAMPLING_RATE
                    channel_name = getattr(h, 'label', f'Channel_{i}')
                
                file_stats['sampling_rates'].append(sampling_rate)
                file_stats['channel_names'].append(channel_name)
                
                if len(sig) > 0:
                    file_stats['signal_stats'].append({
                        'channel': channel_name,
                        'min': np.min(sig),
                        'max': np.max(sig),
                        'mean': np.mean(sig),
                        'std': np.std(sig),
                        'median': np.median(sig),
                        'length': len(sig)
                    })
            
            return file_stats
        except Exception as e:
            return None

    def analyze_patient_original_data(self, patient):
        """分析单个患者的原始数据，内部文件并行处理"""
        patient_path = os.path.join(self.original_path, patient)
        if not os.path.exists(patient_path):
            return None
        
        print(f"正在分析患者 {patient} 的原始数据...")
        
        try:
            patient_files = [f for f in os.listdir(patient_path) if f.endswith('.edf')]
            edf_paths = [os.path.join(patient_path, f) for f in patient_files]
            
            # Parallel processing of EDF files within this patient
            with mp.Pool(mp.cpu_count()) as pool:
                file_results = list(tqdm(
                    pool.imap(self.analyze_edf_file, edf_paths),
                    total=len(edf_paths),
                    desc=f"  {patient} 的EDF文件",
                    leave=False
                ))
            
            # Filter out None results
            valid_results = [result for result in file_results if result is not None]
            
            # Aggregate patient statistics
            patient_stats = {
                'patient': patient,
                'total_files': len(patient_files),
                'processed_files': len(valid_results),
                'total_duration': sum([r['duration'] for r in valid_results]),
                'total_size_mb': sum([r['file_size_mb'] for r in valid_results]),
                'channel_counts': [r['channel_count'] for r in valid_results],
                'sampling_rates': [],
                'channel_names': [],
                'signal_statistics': []
            }
            
            # Aggregate signal statistics
            for result in valid_results:
                patient_stats['sampling_rates'].extend(result['sampling_rates'])
                patient_stats['channel_names'].extend(result['channel_names'])
                patient_stats['signal_statistics'].extend(result['signal_stats'])
            
            # Parse seizure information
            summary_file = os.path.join(patient_path, f"{patient}-summary.txt")
            seizure_info = self._parse_summary_file_comprehensive(summary_file)
            patient_stats['seizure_info'] = seizure_info
            
            return patient_stats
            
        except Exception as e:
            print(f"Error analyzing {patient}: {e}")
            return None

    def analyze_patient_processed_data(self, patient):
        """分析单个患者的处理后数据"""
        patient_path = os.path.join(self.processed_path, patient)
        if not os.path.exists(patient_path):
            return None
        
        print(f"正在分析患者 {patient} 的处理后数据...")
        
        try:
            pkl_files = [f for f in os.listdir(patient_path) if f.endswith('.pkl')]
            
            patient_stats = {
                'patient': patient,
                'total_files': len(pkl_files),
                'signal_statistics': [],
                'channel_names': set(),
                'sampling_rate': self.SAMPLING_RATE,
                'total_duration': 0
            }
            
            # Sample some files for analysis (not all to save time)
            sample_files = pkl_files[:min(5, len(pkl_files))]
            
            for pkl_file in tqdm(sample_files, desc=f"  {patient} 的PKL文件", leave=False):
                try:
                    with open(os.path.join(patient_path, pkl_file), 'rb') as f:
                        data = pickle.load(f)
                    
                    # Analyze each channel's signal
                    for channel in self.channels:
                        if channel in data:
                            sig = data[channel]
                            patient_stats['channel_names'].add(channel)
                            patient_stats['signal_statistics'].append({
                                'channel': channel,
                                'min': np.min(sig),
                                'max': np.max(sig),
                                'mean': np.mean(sig),
                                'std': np.std(sig),
                                'length': len(sig)
                            })
                            
                    # Estimate duration from signal length
                    if len(patient_stats['signal_statistics']) > 0:
                        signal_length = patient_stats['signal_statistics'][0]['length']
                        patient_stats['total_duration'] += signal_length / self.SAMPLING_RATE
                
                except Exception as e:
                    continue
            
            patient_stats['channel_names'] = list(patient_stats['channel_names'])
            return patient_stats
            
        except Exception as e:
            print(f"Error analyzing processed data for {patient}: {e}")
            return None

    def analyze_patient_segmented_data(self, patient_split):
        """分析分割后的数据"""
        split_path = os.path.join(self.segmented_path, patient_split)
        if not os.path.exists(split_path):
            return None
        
        print(f"正在分析 {patient_split} 分割的分段数据...")
        
        try:
            pkl_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
            
            split_stats = {
                'split': patient_split,
                'total_files': len(pkl_files),
                'segment_shapes': [],
                'labels': [],
                'original_signal_stats': [],
                'processed_signal_stats': []
            }
            
            # Sample files for analysis
            sample_files = pkl_files[:min(100, len(pkl_files))]
            
            for pkl_file in tqdm(sample_files, desc=f"  {patient_split} 的分段", leave=False):
                try:
                    with open(os.path.join(split_path, pkl_file), 'rb') as f:
                        data = pickle.load(f)
                    
                    X = data['X']  # Shape: (16, 2560)
                    y = data['y']
                    
                    split_stats['segment_shapes'].append(X.shape)
                    split_stats['labels'].append(y)
                    
                    # Original segment statistics
                    split_stats['original_signal_stats'].append({
                        'shape': X.shape,
                        'min': np.min(X),
                        'max': np.max(X),
                        'mean': np.mean(X),
                        'std': np.std(X),
                        'window_duration': X.shape[1] / self.SAMPLING_RATE,  # 10 seconds
                        'channels': X.shape[0]  # 16 channels
                    })
                    
                    # Simulate the final preprocessing from chb_dataset.py
                    X_resampled = signal.resample(X, 2000, axis=1)  # Resample to 2000
                    X_reshaped = X_resampled.reshape(19, 10, 200)   # Reshape to (19, 10, 200)
                    X_normalized = X_reshaped / 100                  # Normalize
                    
                    split_stats['processed_signal_stats'].append({
                        'original_shape': X.shape,
                        'resampled_shape': X_resampled.shape,
                        'final_shape': X_normalized.shape,
                        'final_min': np.min(X_normalized),
                        'final_max': np.max(X_normalized),
                        'final_mean': np.mean(X_normalized),
                        'final_std': np.std(X_normalized),
                        'final_sampling_rate': 2000 / 10,  # 200 Hz effective
                        'final_channels': 19,
                        'final_window_duration': 10  # seconds
                    })
                    
                except Exception as e:
                    continue
            
            return split_stats
            
        except Exception as e:
            print(f"Error analyzing segmented data for {patient_split}: {e}")
            return None

    def analyze_comprehensive_dataset(self):
        """使用顺序患者处理和内部并行化的综合分析"""
        print("=== CHB-MIT数据集综合分析 ===")
        
        # 1. 分析原始数据（患者顺序处理，每个患者内文件并行处理）
        print("\n1. 分析原始数据...")
        available_patients = [f"chb{param[0]}" for param in self.process1_params]
        
        original_results = []
        for patient in tqdm(available_patients, desc="处理患者（原始数据）"):
            result = self.analyze_patient_original_data(patient)
            if result:
                original_results.append(result)
        
        # 2. 分析处理后数据
        print("\n2. 分析处理后数据...")
        processed_results = []
        if os.path.exists(self.processed_path):
            for patient in tqdm(available_patients, desc="处理患者（处理后数据）"):
                result = self.analyze_patient_processed_data(patient)
                if result:
                    processed_results.append(result)
        
        # 3. 分析分段数据
        print("\n3. 分析分段数据...")
        segmented_results = []
        if os.path.exists(self.segmented_path):
            for split in ['train', 'val', 'test']:
                result = self.analyze_patient_segmented_data(split)
                if result:
                    segmented_results.append(result)
        
        return self._aggregate_comprehensive_stats(original_results, processed_results, segmented_results)

    def _aggregate_comprehensive_stats(self, original_results, processed_results, segmented_results):
        """Aggregate statistics from all analysis results"""
        
        # Aggregate original data statistics
        original_stats = {
            'total_patients': len(original_results),
            'total_files': sum([r['total_files'] for r in original_results]),
            'total_duration_hours': sum([r['total_duration'] for r in original_results]) / 3600,
            'total_size_gb': sum([r['total_size_mb'] for r in original_results]) / 1024,
            'sampling_rate_distribution': Counter(),
            'channel_distribution': Counter(),
            'signal_amplitude_stats': {
                'all_minimums': [],
                'all_maximums': [],
                'all_means': [],
                'all_stds': []
            },
            'seizure_summary': {
                'total_seizures': 0,
                'patients_with_seizures': 0,
                'total_seizure_duration': 0
            }
        }
        
        for result in original_results:
            original_stats['sampling_rate_distribution'].update(result['sampling_rates'])
            original_stats['channel_distribution'].update(result['channel_names'])
            
            for sig_stat in result['signal_statistics']:
                original_stats['signal_amplitude_stats']['all_minimums'].append(sig_stat['min'])
                original_stats['signal_amplitude_stats']['all_maximums'].append(sig_stat['max'])
                original_stats['signal_amplitude_stats']['all_means'].append(sig_stat['mean'])
                original_stats['signal_amplitude_stats']['all_stds'].append(sig_stat['std'])
            
            seizure_info = result['seizure_info']
            original_stats['seizure_summary']['total_seizures'] += seizure_info['total_seizures']
            if seizure_info['total_seizures'] > 0:
                original_stats['seizure_summary']['patients_with_seizures'] += 1
            original_stats['seizure_summary']['total_seizure_duration'] += sum(seizure_info['seizure_durations'])
        
        # Aggregate processed data statistics
        processed_stats = {
            'total_patients': len(processed_results),
            'total_files': sum([r['total_files'] for r in processed_results]),
            'channels': len(self.channels),
            'sampling_rate': self.SAMPLING_RATE,
            'signal_amplitude_stats': {
                'all_minimums': [],
                'all_maximums': [],
                'all_means': [],
                'all_stds': []
            }
        }
        
        for result in processed_results:
            for sig_stat in result['signal_statistics']:
                processed_stats['signal_amplitude_stats']['all_minimums'].append(sig_stat['min'])
                processed_stats['signal_amplitude_stats']['all_maximums'].append(sig_stat['max'])
                processed_stats['signal_amplitude_stats']['all_means'].append(sig_stat['mean'])
                processed_stats['signal_amplitude_stats']['all_stds'].append(sig_stat['std'])
        
        # Aggregate segmented data statistics
        segmented_stats = {
            'splits': {},
            'total_segments': 0,
            'label_distribution': Counter(),
            'original_segment_stats': {
                'window_duration': 10,  # seconds
                'channels': 16,
                'sampling_rate': self.SAMPLING_RATE,
                'amplitude_stats': {'mins': [], 'maxs': [], 'means': [], 'stds': []}
            },
            'final_segment_stats': {
                'window_duration': 10,  # seconds
                'channels': 19,
                'sampling_rate': 200,  # effective Hz
                'shape': '(19, 10, 200)',
                'amplitude_stats': {'mins': [], 'maxs': [], 'means': [], 'stds': []}
            }
        }
        
        for result in segmented_results:
            split_name = result['split']
            segmented_stats['splits'][split_name] = {
                'total_files': result['total_files'],
                'labels': Counter(result['labels'])
            }
            segmented_stats['total_segments'] += result['total_files']
            segmented_stats['label_distribution'].update(result['labels'])
            
            # Aggregate signal statistics
            for orig_stat in result['original_signal_stats']:
                segmented_stats['original_segment_stats']['amplitude_stats']['mins'].append(orig_stat['min'])
                segmented_stats['original_segment_stats']['amplitude_stats']['maxs'].append(orig_stat['max'])
                segmented_stats['original_segment_stats']['amplitude_stats']['means'].append(orig_stat['mean'])
                segmented_stats['original_segment_stats']['amplitude_stats']['stds'].append(orig_stat['std'])
            
            for proc_stat in result['processed_signal_stats']:
                segmented_stats['final_segment_stats']['amplitude_stats']['mins'].append(proc_stat['final_min'])
                segmented_stats['final_segment_stats']['amplitude_stats']['maxs'].append(proc_stat['final_max'])
                segmented_stats['final_segment_stats']['amplitude_stats']['means'].append(proc_stat['final_mean'])
                segmented_stats['final_segment_stats']['amplitude_stats']['stds'].append(proc_stat['final_std'])
        
        return original_stats, processed_stats, segmented_stats

    def _parse_summary_file_comprehensive(self, summary_path):
        """Comprehensive parsing of summary files"""
        seizure_info = {
            'file_seizure_counts': [],
            'seizure_durations': [],
            'total_seizures': 0,
            'files_with_seizures': 0
        }
        
        if not os.path.exists(summary_path):
            return seizure_info
            
        try:
            with open(summary_path, 'r') as f:
                lines = f.readlines()
                
            current_file_seizures = 0
            for i in range(len(lines)):
                line = lines[i].split()
                if len(line) >= 3 and line[2].endswith('.edf'):
                    # Reset for new file
                    current_file_seizures = 0
                    
                    # Find seizure count for this file
                    j = i + 1
                    while j < len(lines):
                        if lines[j].split() and len(lines[j].split()) > 0 and lines[j].split()[0] == "Number":
                            try:
                                current_file_seizures = int(lines[j].split()[-1])
                                seizure_info['file_seizure_counts'].append(current_file_seizures)
                                seizure_info['total_seizures'] += current_file_seizures
                                if current_file_seizures > 0:
                                    seizure_info['files_with_seizures'] += 1
                                break
                            except:
                                pass
                        j += 1
                        
                    # Get seizure durations if seizures > 0
                    if current_file_seizures > 0:
                        j = i + 1
                        seizure_count = 0
                        while j < len(lines) and seizure_count < current_file_seizures:
                            l = lines[j].split()
                            if len(l) > 0 and l[0] == "Seizure" and "Start" in l:
                                try:
                                    start = int(l[-2])
                                    if j + 1 < len(lines):
                                        end = int(lines[j + 1].split()[-2])
                                        duration = end - start
                                        seizure_info['seizure_durations'].append(duration)
                                        seizure_count += 1
                                except:
                                    pass
                            j += 1
        except Exception as e:
            print(f"Error parsing summary file {summary_path}: {e}")
                    
        return seizure_info
    
    def analyze_preprocessing_pipeline(self):
        """Analyze preprocessing pipeline from code"""
        print("\n=== ANALYZING PREPROCESSING PIPELINE ===")
        
        # Calculate expected output from process2.py
        total_files_processed = len([p for p in self.process1_params])
        
        # Estimate segments per patient based on process1.py parameters
        estimated_segments = {
            'train': 0,
            'val': 0,
            'test': 0
        }
        
        estimated_seizure_segments = 0
        estimated_normal_segments = 0
        
        for patient_param in self.process1_params:
            patient = f"chb{patient_param[0]}"
            start_file = patient_param[2]
            end_file = patient_param[3]
            num_files = end_file - start_file + 1
            
            # Estimate 1 hour average per file, 360 segments per hour (10-second segments)
            estimated_segments_per_patient = num_files * 360
            
            if patient in self.test_pats:
                estimated_segments['test'] += estimated_segments_per_patient
            elif patient in self.val_pats:
                estimated_segments['val'] += estimated_segments_per_patient
            else:
                estimated_segments['train'] += estimated_segments_per_patient
        
        # Processing steps analysis
        pipeline_stats = {
            'stage1_process1': {
                'input_format': 'EDF files',
                'output_format': 'PKL files',
                'channel_selection': len(self.channels),
                'patients_processed': len(self.process1_params),
                'compression': 'pickle',
                'metadata_extraction': True
            },
            'stage2_process2': {
                'segmentation': '10-second windows',
                'segment_length_samples': 10 * self.SAMPLING_RATE,
                'overlap': 'Non-overlapping + seizure augmentation',
                'augmentation': '5-second sliding window for seizures',
                'labeling': 'Binary (seizure/non-seizure)',
                'estimated_total_segments': sum(estimated_segments.values())
            },
            'stage3_chb_dataset': {
                'resampling': f'{10 * self.SAMPLING_RATE} → 2000 samples',
                'reshape': 'From (16, 2000) to (19, 10, 200)',
                'channel_padding': '16 → 19 channels (zero padding)',
                'normalization': 'Division by 100',
                'final_shape': '(19, 10, 200)',
                'effective_sampling_rate': '200 Hz'
            }
        }
        
        return estimated_segments, pipeline_stats
    
    def _summarize_comprehensive_stats(self, stats):
        """Comprehensive summary of original dataset"""
        # Channel analysis
        channel_stats = {}
        for channel_name, channel_data in stats['signal_statistics_per_channel'].items():
            if channel_data:
                channel_stats[channel_name] = {
                    'count': len(channel_data),
                    'avg_amplitude_range': np.mean([d['max'] - d['min'] for d in channel_data]),
                    'avg_mean': np.mean([d['mean'] for d in channel_data]),
                    'avg_std': np.mean([d['std'] for d in channel_data])
                }
        
        summary = {
            'dataset_overview': {
                'total_patients': len(stats['patients']),
                'total_files': stats['total_files'],
                'avg_files_per_patient': np.mean(stats['files_per_patient']) if stats['files_per_patient'] else 0,
                'total_recording_hours': sum(stats['recording_hours']),
                'avg_recording_hours_per_patient': np.mean(stats['recording_hours']) if stats['recording_hours'] else 0
            },
            'technical_specs': {
                'sampling_rate': self.SAMPLING_RATE,
                'common_sampling_rate': Counter(stats['sampling_rates']).most_common(1)[0] if stats['sampling_rates'] else (None, 0),
                'avg_duration_per_file': np.mean(stats['durations']) if stats['durations'] else 0,
                'avg_channels_per_file': np.mean(stats['channel_counts']) if stats['channel_counts'] else 0,
                'avg_file_size_mb': np.mean(stats['file_sizes']) if stats['file_sizes'] else 0
            },
            'channel_analysis': {
                'unique_channels': len(set(stats['channel_names'])),
                'most_common_channels': Counter(stats['channel_names']).most_common(20),
                'target_channels': self.channels,
                'target_channel_count': len(self.channels)
            },
            'signal_characteristics': {
                'overall_amplitude_range': {
                    'min': np.min(stats['signal_amplitudes']) if stats['signal_amplitudes'] else 0,
                    'max': np.max(stats['signal_amplitudes']) if stats['signal_amplitudes'] else 0,
                    'mean': np.mean(stats['signal_amplitudes']) if stats['signal_amplitudes'] else 0,
                    'std': np.std(stats['signal_amplitudes']) if stats['signal_amplitudes'] else 0
                },
                'per_channel_stats': channel_stats
            },
            'seizure_analysis': {
                'total_seizures': sum(stats['seizure_counts']),
                'files_with_seizures': len([x for x in stats['seizure_counts'] if x > 0]),
                'files_without_seizures': len([x for x in stats['seizure_counts'] if x == 0]),
                'avg_seizures_per_file': np.mean(stats['seizure_counts']) if stats['seizure_counts'] else 0,
                'seizure_duration_stats': {
                    'avg_duration': np.mean(stats['seizure_durations']) if stats['seizure_durations'] else 0,
                    'min_duration': np.min(stats['seizure_durations']) if stats['seizure_durations'] else 0,
                    'max_duration': np.max(stats['seizure_durations']) if stats['seizure_durations'] else 0,
                    'total_seizure_time': sum(stats['seizure_durations']) if stats['seizure_durations'] else 0
                }
            }
        }
        return summary
    
    def generate_comprehensive_documentation(self):
        """生成完整文档"""
        print("正在生成CHB-MIT数据集综合文档...")
        
        # 分析所有数据阶段
        original_stats, processed_stats, segmented_stats = self.analyze_comprehensive_dataset()
        
        # 从代码分析预处理流水线
        estimated_segments, pipeline_stats = self.analyze_preprocessing_pipeline()
        
        # 生成文档
        doc_content = self._create_detailed_markdown_doc(original_stats, processed_stats, segmented_stats, pipeline_stats)
        
        # 保存文档
        with open(os.path.join(self.output_dir, 'CHB_MIT_详细分析.md'), 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        # 生成可视化图表
        self._create_detailed_visualizations(original_stats, processed_stats, segmented_stats)
        
        print(f"详细文档已保存到 {self.output_dir}")
        return original_stats, processed_stats, segmented_stats, pipeline_stats

    def _create_detailed_markdown_doc(self, original_stats, processed_stats, segmented_stats, pipeline_stats):
        """创建详细的中文markdown文档"""
        
        # Calculate summary statistics
        def calc_stats(values):
            if not values:
                return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'median': 0}
            return {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            }
        
        orig_amp_stats = calc_stats(original_stats['signal_amplitude_stats']['all_means'])
        proc_amp_stats = calc_stats(processed_stats['signal_amplitude_stats']['all_means'])
        
        doc = f"""# CHB-MIT数据集详细分析报告

## 执行摘要

本综合分析检查了CHB-MIT数据集在所有处理阶段的详细统计信息。

### 数据集概览
- **原始患者数**: {original_stats['total_patients']}
- **原始文件数**: {original_stats['total_files']}
- **总录制时间**: {original_stats['total_duration_hours']:.1f} 小时
- **总数据大小**: {original_stats['total_size_gb']:.1f} GB
- **总癫痫发作次数**: {original_stats['seizure_summary']['total_seizures']}

## 1. 原始数据分析

### 基本统计信息
- **分析患者数**: {original_stats['total_patients']}
- **EDF文件总数**: {original_stats['total_files']}
- **总录制时长**: {original_stats['total_duration_hours']:.1f} 小时
- **数据集总大小**: {original_stats['total_size_gb']:.1f} GB
- **每患者平均**: {original_stats['total_duration_hours']/max(original_stats['total_patients'], 1):.1f} 小时

### 采样率分布
"""
        
        for rate, count in original_stats['sampling_rate_distribution'].most_common(5):
            doc += f"- {rate} Hz: {count} 个通道 ({count/sum(original_stats['sampling_rate_distribution'].values())*100:.1f}%)\n"
        
        doc += f"""

### 信号幅度统计（原始）
- **范围**: {orig_amp_stats['min']:.2f} 到 {orig_amp_stats['max']:.2f} µV
- **均值**: {orig_amp_stats['mean']:.2f} µV
- **标准差**: {orig_amp_stats['std']:.2f} µV
- **中位数**: {orig_amp_stats['median']:.2f} µV

### 通道分布（前15名）
"""
        
        for i, (channel, count) in enumerate(original_stats['channel_distribution'].most_common(15)):
            doc += f"{i+1:2d}. {channel}: {count} 次出现\n"
        
        doc += f"""

### 癫痫发作分析
- **总发作次数**: {original_stats['seizure_summary']['total_seizures']}
- **有发作的患者**: {original_stats['seizure_summary']['patients_with_seizures']}/{original_stats['total_patients']}
- **总发作时长**: {original_stats['seizure_summary']['total_seizure_duration']:.1f} 秒
- **发作率**: {original_stats['seizure_summary']['total_seizure_duration']/3600/max(original_stats['total_duration_hours'], 1)*100:.2f}% 的录制时间

## 2. 处理后数据分析

### 处理统计信息
- **处理的患者数**: {processed_stats['total_patients']}
- **PKL文件总数**: {processed_stats['total_files']}
- **目标通道数**: {processed_stats['channels']} (标准化)
- **采样率**: {processed_stats['sampling_rate']} Hz (保持不变)

### 信号幅度统计（通道选择后）
- **范围**: {proc_amp_stats['min']:.2f} 到 {proc_amp_stats['max']:.2f} µV
- **均值**: {proc_amp_stats['mean']:.2f} µV
- **标准差**: {proc_amp_stats['std']:.2f} µV
- **中位数**: {proc_amp_stats['median']:.2f} µV

### 目标通道（来自process2.py）
"""
        
        for i, channel in enumerate(self.channels, 1):
            doc += f"{i:2d}. {channel}\n"
        
        doc += f"""

## 3. 分段数据分析

### 数据集划分
"""
        
        for split_name, split_data in segmented_stats['splits'].items():
            split_names = {'train': '训练', 'val': '验证', 'test': '测试'}
            doc += f"- **{split_names.get(split_name, split_name)}**: {split_data['total_files']:,} 个分段\n"
        
        doc += f"- **总计**: {segmented_stats['total_segments']:,} 个分段\n\n"
        
        doc += f"""### 标签分布
- **非癫痫(0)**: {segmented_stats['label_distribution'].get(0, 0):,} 个分段 ({segmented_stats['label_distribution'].get(0, 0)/max(segmented_stats['total_segments'], 1)*100:.1f}%)
- **癫痫(1)**: {segmented_stats['label_distribution'].get(1, 0):,} 个分段 ({segmented_stats['label_distribution'].get(1, 0)/max(segmented_stats['total_segments'], 1)*100:.1f}%)

### 原始分段特征（最终处理前）
- **窗口时长**: {segmented_stats['original_segment_stats']['window_duration']} 秒
- **通道数**: {segmented_stats['original_segment_stats']['channels']}
- **采样率**: {segmented_stats['original_segment_stats']['sampling_rate']} Hz
- **每分段样本数**: {segmented_stats['original_segment_stats']['window_duration'] * segmented_stats['original_segment_stats']['sampling_rate']}

#### 原始分段幅度统计
"""
        
        if segmented_stats['original_segment_stats']['amplitude_stats']['means']:
            orig_seg_stats = calc_stats(segmented_stats['original_segment_stats']['amplitude_stats']['means'])
            doc += f"""- **范围**: {orig_seg_stats['min']:.2f} 到 {orig_seg_stats['max']:.2f} µV
- **均值**: {orig_seg_stats['mean']:.2f} µV
- **标准差**: {orig_seg_stats['std']:.2f} µV
"""
        
        doc += f"""

### 最终处理特征（chb_dataset.py后）
- **最终形状**: {segmented_stats['final_segment_stats']['shape']}
- **有效采样率**: {segmented_stats['final_segment_stats']['sampling_rate']} Hz
- **通道数**: {segmented_stats['final_segment_stats']['channels']} (从{segmented_stats['original_segment_stats']['channels']}填充)
- **窗口时长**: {segmented_stats['final_segment_stats']['window_duration']} 秒

#### 最终幅度统计（归一化后）
"""
        
        if segmented_stats['final_segment_stats']['amplitude_stats']['means']:
            final_seg_stats = calc_stats(segmented_stats['final_segment_stats']['amplitude_stats']['means'])
            doc += f"""- **范围**: {final_seg_stats['min']:.3f} 到 {final_seg_stats['max']:.3f} (归一化)
- **均值**: {final_seg_stats['mean']:.3f} (归一化)
- **标准差**: {final_seg_stats['std']:.3f} (归一化)
"""
        
        doc += f"""

## 4. 处理流水线总结

### 第一阶段: EDF → PKL (process1.py)
- **输入**: {original_stats['total_files']} 个EDF文件
- **输出**: {processed_stats['total_files']} 个PKL文件
- **通道选择**: {len(original_stats['channel_distribution'])} → {processed_stats['channels']} 个通道
- **压缩**: Pickle格式带元数据

### 第二阶段: PKL → 分段 (process2.py)
- **输入**: 连续信号的PKL文件
- **分段**: {segmented_stats['original_segment_stats']['window_duration']}秒非重叠窗口
- **增强**: 癫痫期间5秒滑动窗口
- **输出**: {segmented_stats['total_segments']:,} 个分段
- **标注**: 二进制 (0=正常, 1=癫痫)

### 第三阶段: 最终处理 (chb_dataset.py)
- **重采样**: {segmented_stats['original_segment_stats']['window_duration'] * segmented_stats['original_segment_stats']['sampling_rate']} → 2000 样本
- **通道填充**: {segmented_stats['original_segment_stats']['channels']} → {segmented_stats['final_segment_stats']['channels']} 通道
- **重塑**: (16, 2000) → {segmented_stats['final_segment_stats']['shape']}
- **归一化**: 除以100

## 5. 数据变换总结

```
原始EDF:
├─ 采样率: {list(original_stats['sampling_rate_distribution'].keys())[0] if original_stats['sampling_rate_distribution'] else 'N/A'} Hz
├─ 通道: {len(original_stats['channel_distribution'])} 种独特类型
├─ 幅度: {orig_amp_stats['mean']:.1f} ± {orig_amp_stats['std']:.1f} µV
└─ 时长: {original_stats['total_duration_hours']:.1f} 小时

处理后PKL:
├─ 采样率: {processed_stats['sampling_rate']} Hz (保持)
├─ 通道: {processed_stats['channels']} (标准化)
├─ 幅度: {proc_amp_stats['mean']:.1f} ± {proc_amp_stats['std']:.1f} µV
└─ 格式: 带元数据的Pickle

分段:
├─ 窗口大小: {segmented_stats['original_segment_stats']['window_duration']}秒 × {segmented_stats['original_segment_stats']['channels']} 通道
├─ 样本: 每分段{segmented_stats['original_segment_stats']['window_duration'] * segmented_stats['original_segment_stats']['sampling_rate']}个
├─ 总分段数: {segmented_stats['total_segments']:,}
└─ 标签: {segmented_stats['label_distribution'].get(1, 0):,} 癫痫, {segmented_stats['label_distribution'].get(0, 0):,} 正常

最终张量:
├─ 形状: {segmented_stats['final_segment_stats']['shape']}
├─ 采样率: {segmented_stats['final_segment_stats']['sampling_rate']} Hz (有效)
├─ 幅度: 归一化 (÷100)
└─ 内存: ~{segmented_stats['total_segments'] * 19 * 10 * 200 * 4 / 1024 / 1024 / 1024:.1f} GB
```

## 6. 关键洞察

### 数据质量
1. **采样一致性**: {list(original_stats['sampling_rate_distribution'].most_common(1))[0][1]/sum(original_stats['sampling_rate_distribution'].values())*100:.1f}% 的通道使用主要采样率
2. **通道标准化**: 通道变异性从{len(original_stats['channel_distribution'])}种减少到{processed_stats['channels']}种类型
3. **癫痫表示**: {segmented_stats['label_distribution'].get(1, 0)/max(segmented_stats['total_segments'], 1)*100:.1f}% 癫痫分段 (类别不平衡)

### 处理影响
1. **数据缩减**: {segmented_stats['final_segment_stats']['sampling_rate']}/{segmented_stats['original_segment_stats']['sampling_rate']} = {segmented_stats['final_segment_stats']['sampling_rate']/segmented_stats['original_segment_stats']['sampling_rate']:.2f}× 采样率缩减
2. **通道填充**: 通过零填充增加{segmented_stats['final_segment_stats']['channels'] - segmented_stats['original_segment_stats']['channels']}个通道
3. **幅度归一化**: 信号缩放到神经网络友好范围

### 建议
1. **类别不平衡**: 使用加权损失函数或数据增强
2. **内存优化**: 对大数据集考虑批量加载
3. **验证**: 按患者划分防止数据泄露
4. **监控**: 跟踪每通道统计以进行质量控制
"""
        
        return doc

    def _create_detailed_visualizations(self, original_stats, processed_stats, segmented_stats):
        """创建详细的可视化图表"""
        plt.style.use('default')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(24, 18))
        
        # Plot 1: 采样率分布
        ax1 = plt.subplot(4, 4, 1)
        if original_stats['sampling_rate_distribution']:
            rates, counts = zip(*original_stats['sampling_rate_distribution'].most_common(5))
            ax1.bar(range(len(rates)), counts, alpha=0.7)
            ax1.set_title('采样率分布（原始）')
            ax1.set_xlabel('采样率 (Hz)')
            ax1.set_ylabel('通道数')
            ax1.set_xticks(range(len(rates)))
            ax1.set_xticklabels([f'{r}' for r in rates])
        
        # Plot 2: 通道分布
        ax2 = plt.subplot(4, 4, 2)
        if original_stats['channel_distribution']:
            channels, counts = zip(*original_stats['channel_distribution'].most_common(10))
            ax2.barh(range(len(channels)), counts, alpha=0.7)
            ax2.set_title('前10个通道（原始）')
            ax2.set_xlabel('频率')
            ax2.set_yticks(range(len(channels)))
            ax2.set_yticklabels(channels, fontsize=8)
        
        # Plot 3: 幅度分布比较
        ax3 = plt.subplot(4, 4, 3)
        stages = ['原始', '处理后', '最终']
        if (original_stats['signal_amplitude_stats']['all_means'] and 
            processed_stats['signal_amplitude_stats']['all_means'] and 
            segmented_stats['final_segment_stats']['amplitude_stats']['means']):
            
            means = [
                np.mean(original_stats['signal_amplitude_stats']['all_means']),
                np.mean(processed_stats['signal_amplitude_stats']['all_means']),
                np.mean(segmented_stats['final_segment_stats']['amplitude_stats']['means'])
            ]
            ax3.plot(stages, means, 'o-', linewidth=2, markersize=8)
            ax3.set_title('各阶段信号幅度')
            ax3.set_ylabel('平均幅度')
            ax3.set_yscale('log')
        
        # Plot 4: 数据集划分
        ax4 = plt.subplot(4, 4, 4)
        if segmented_stats['splits']:
            splits = list(segmented_stats['splits'].keys())
            split_names = {'train': '训练', 'val': '验证', 'test': '测试'}
            labels = [split_names.get(s, s) for s in splits]
            counts = [segmented_stats['splits'][s]['total_files'] for s in splits]
            ax4.pie(counts, labels=labels, autopct='%1.1f%%')
            ax4.set_title('数据集划分分布')
        
        # Plot 5: 标签分布
        ax5 = plt.subplot(4, 4, 5)
        if segmented_stats['label_distribution']:
            labels = ['正常', '癫痫']
            counts = [segmented_stats['label_distribution'].get(0, 0), 
                     segmented_stats['label_distribution'].get(1, 0)]
            colors = ['lightblue', 'red']
            ax5.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
            ax5.set_title('标签分布')
        
        # Plot 6: 处理流水线流程
        ax6 = plt.subplot(4, 4, 6)
        stages = ['原始\nEDF', '处理后\nPKL', '分段\n文件', '最终\n张量']
        values = [
            original_stats['total_files'],
            processed_stats['total_files'],
            segmented_stats['total_segments'],
            segmented_stats['total_segments']
        ]
        ax6.plot(stages, values, 'o-', linewidth=2, markersize=8)
        ax6.set_title('数据处理流程')
        ax6.set_ylabel('数量')
        ax6.tick_params(axis='x', rotation=45)
        
        # Plot 7: 癫痫分析
        ax7 = plt.subplot(4, 4, 7)
        seizure_data = [
            original_stats['seizure_summary']['patients_with_seizures'],
            original_stats['total_patients'] - original_stats['seizure_summary']['patients_with_seizures']
        ]
        ax7.pie(seizure_data, labels=['有癫痫', '无癫痫'], autopct='%1.1f%%')
        ax7.set_title('按癫痫存在划分的患者')
        
        # Plot 8: 信号形状变换
        ax8 = plt.subplot(4, 4, 8)
        shapes = ['原始\n(可变)', '分段\n(16×2560)', '最终\n(19×10×200)']
        complexities = [1, 16*2560, 19*10*200]  # 相对复杂度
        ax8.bar(shapes, complexities, alpha=0.7, color=['blue', 'orange', 'green'])
        ax8.set_title('信号形状变换')
        ax8.set_ylabel('每样本数据点')
        ax8.set_yscale('log')
        
        # Plot 9-12: Detailed statistics plots
        # ... (additional plots for detailed statistics)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chb_mit_详细分析.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("详细可视化图表已保存到 chb_mit_详细分析.png")

# Usage example
if __name__ == "__main__":
    analyzer = CHBMITDatasetAnalyzer(r"D:\datasets\eeg\dataset_dir_original\chb-mit-scalp-eeg-database-1.0.0")
    original_stats, processed_stats, segmented_stats, pipeline_stats = analyzer.generate_comprehensive_documentation()
    
    print("\n" + "="*60)
    print("CHB-MIT数据集详细分析完成")
    print("="*60)
    print(f"📁 结果保存到: {analyzer.output_dir}")
    print(f"📊 原始患者数: {original_stats['total_patients']}")
    print(f"📂 原始文件数: {original_stats['total_files']}")
    print(f"⏱️ 总录制时间: {original_stats['total_duration_hours']:.1f} 小时")
    print(f"💾 总数据大小: {original_stats['total_size_gb']:.1f} GB")
    print(f"🧠 总癫痫发作: {original_stats['seizure_summary']['total_seizures']}")
    
    if segmented_stats['total_segments'] > 0:
        seizure_ratio = segmented_stats['label_distribution'].get(1, 0) / segmented_stats['total_segments'] * 100
        print(f"📈 总分段数: {segmented_stats['total_segments']:,}")
        print(f"⚖️ 癫痫比例: {seizure_ratio:.2f}%")
    else:
        print(f"📈 估计分段数: {sum(estimated_segments.values()):,}")
        print(f"⚖️ 癫痫比例: 基于代码分析估计")
    
    print(f"🔧 处理阶段已分析: 3个完整阶段")
    print("="*60)
    
    # 生成分析完成的总结
    print("\n📝 分析总结:")
    if original_stats['total_files'] > 0:
        print("✅ 原始EDF文件: 成功分析，使用并行处理")
        print(f"   - 发现 {original_stats['total_files']} 个文件来自 {original_stats['total_patients']} 个患者")
        print(f"   - 总录制时间: {original_stats['total_duration_hours']:.1f} 小时")
        print(f"   - 癫痫事件: {original_stats['seizure_summary']['total_seizures']}")
        print(f"   - 使用 {mp.cpu_count()} 个CPU核心进行并行处理")
    else:
        print("❌ 原始EDF文件: 未找到或无法访问")
        print("   - 分析仅基于代码结构")
    
    print("✅ 预处理流水线: 从代码分析")
    print(f"   - 估计输出: {sum(estimated_segments.values()):,} 个分段")
    print(f"   - 内存估计: ~{sum(estimated_segments.values())*19*10*200*4/1024/1024/1024:.1f} GB")
    
    print("✅ 文档: 已生成")
    print(f"   - Markdown报告: {analyzer.output_dir}/CHB_MIT_详细分析.md")
    print(f"   - 可视化图表: {analyzer.output_dir}/chb_mit_详细分析.png")
