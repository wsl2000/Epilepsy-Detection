# -*- coding: utf-8 -*-
"""
此文件不应被修改，由我们提供并会被重置。

脚本用于测试预训练模型

@author: Maurice Rohr
"""

from predict import predict_labels
from wettbewerb import EEGDataset, save_predictions
import argparse
import time
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用给定模型进行预测')
    parser.add_argument('--test_dir', action='store', type=str, default=r'D:\datasets\eeg\dataset_dir_original\shared_data\training', help='测试数据文件夹路径')
    parser.add_argument('--model_name', action='store', type=str, default='model.json', help='模型文件名')
    parser.add_argument('--allow_fail', action='store_true', default=False, help='是否允许失败')
    args = parser.parse_args()
    
    # 从文件夹创建EEG数据集
    dataset = EEGDataset(args.test_dir)
    print(f"正在测试模型，共有 {len(dataset)} 条记录")
    
    predictions = list()
    start_time = time.time()
    
    # 对数据集中的每个元素（记录）调用预测方法
    for item in tqdm(dataset, desc="预测进度", unit="条"):
        id, channels, data, fs, ref_system, eeg_label = item
        try:
            _prediction = predict_labels(channels, data, fs, ref_system, model_name=args.model_name)
            _prediction["id"] = id
            predictions.append(_prediction)
        except Exception as e:
            print(f"预测失败: {e}, ID: {id}")
            if args.allow_fail:
                raise
    pred_time = time.time() - start_time        
    
    save_predictions(predictions) # 将预测结果保存到CSV文件
    print("运行时间", pred_time, "秒")
