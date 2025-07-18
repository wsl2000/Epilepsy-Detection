import os
import sys
import pandas as pd
import numpy as np
import argparse
from typing import Tuple, List, Dict, Union

def score(test_dir: str = '../test/') -> Dict[str, Union[float, List[int]]]:
    """
    计算竞赛相关的指标，以及其他指标
    参数
    ----------
    folder : str, optional
        测试数据的位置。默认值 '../test'。

    返回
    -------
    字典，包含以下键：
        performance_metric_WS23 : float
            用于排名的指标。F1分数，仅对预测在正确区间的情况
        performance_metric_SS23 : float
            发作起始点检测的MAE，带有错误分类的惩罚项
        F1 : float
            癫痫发作分类的F1分数（Seizure present = 正类）
        sensitivity :  float
            癫痫发作分类的灵敏度
        PPV : float
            癫痫发作分类的阳性预测值
        detection_error_onset : float
            起始点检测的平均绝对误差（每条记录有上限）
        detection_error_offset : float
            结束点检测的平均绝对误差（每条记录有上限）
        confusion_matrix : List[int]
            癫痫发作分类的混淆矩阵 [TP,FN,FP,TN]
    """

    if not os.path.exists("PREDICTIONS.csv"):
        sys.exit("没有预测结果")  

    if not os.path.exists(os.path.join(test_dir, "REFERENCE.csv")):
        sys.exit("没有真实标签")  

    df_pred = pd.read_csv("PREDICTIONS.csv")   # 预测分类
    df_gt = pd.read_csv(os.path.join(test_dir, "REFERENCE.csv"), sep=',', header=None)  # 真实标签

    ONSET_PENALTY = 60 # 秒，如果未检测到发作或误差大于惩罚值，则记为惩罚值
    FALSE_CLASSIFICATION_PENALTY = 60 # 秒，若错误检测到发作，则加惩罚秒数
    INTERVAL_DELAY = 30 # 秒，预测发作时间与真实时间允许的最大误差

    # 确保每个真实标签都有预测
    gt_seizure_present, gt_onset, gt_offset, pred_seizure_present, pred_onset, pred_offset, pred_seizure_confidence, pred_onset_confidence, pred_offset_confidence = match_predictions(df_gt, df_pred)
    # 计算分类问题的混淆矩阵
    confusion_matrix = compute_confusion_matrix(gt_seizure_present, pred_seizure_present)
    # 基于TP,FN,FP,TN计算基本指标
    sensitivity, PPV, F1, accuracy = compute_basic_metrics(*confusion_matrix)
    # 计算起始点预测误差（延迟）
    detection_error_onset = compute_detection_error(gt_seizure_present, pred_seizure_present, gt_onset, pred_onset, ONSET_PENALTY)
    # 计算结束点预测误差（可选）
    detection_error_offset = compute_detection_error(gt_seizure_present, pred_seizure_present, gt_offset, pred_offset, ONSET_PENALTY)
    # 计算加权指标（2023年夏季学期）（越低越好）
    performance_metric_SS23 = compute_SS23_performance_metric(detection_error_onset, confusion_matrix, FALSE_CLASSIFICATION_PENALTY)
    i_score_sensitivity, i_score_PPV, i_score_F1, i_score_accuracy = compute_interval_scores(gt_seizure_present, pred_seizure_present, gt_onset, pred_onset, max_delay=INTERVAL_DELAY)
    # 计算基于区间的指标（2023年冬季学期）（越高越好）
    performance_metric_WS23 = compute_WS23_performance_metric(gt_seizure_present, pred_seizure_present, gt_onset, pred_onset, max_delay=INTERVAL_DELAY)

    metrics = {
        "performance_metric_WS23": performance_metric_WS23,
        "performance_metric_SS23": performance_metric_SS23,
        "detection_error_onset": detection_error_onset,
        "F1": F1,
        "sensitivity": sensitivity,
        "PPV": PPV,
        "accuracy": accuracy,
        "detection_error_offset": detection_error_offset,
        "i_sensitivity": i_score_sensitivity,
        "i_PPV": i_score_PPV,
        "i_accuracy": i_score_accuracy,
        "confusion_matrix": confusion_matrix
    }

    return metrics

def match_predictions(df_gt: pd.DataFrame, df_pred: pd.DataFrame):
    N_files = df_gt.shape[0]
    gt_seizure_present = df_gt[1].to_numpy().astype(bool)
    gt_onset = df_gt[2].to_numpy()
    gt_offset = df_gt[3].to_numpy()
    pred_seizure_present = np.zeros_like(gt_seizure_present, dtype=bool)
    pred_onset = np.zeros_like(gt_onset)
    pred_offset = np.zeros_like(gt_offset)
    pred_seizure_confidence = np.zeros_like(gt_onset)
    pred_onset_confidence = np.zeros_like(gt_onset)
    pred_offset_confidence = np.zeros_like(gt_onset)
    for i in range(N_files):
        _gt_name = df_gt[0][i]
        pred_indx = df_pred[df_pred['id'] == _gt_name].index.values

        if not pred_indx.size:
            print("缺少 " + _gt_name + " 的预测，假定为“无发作”。")
            pred_seizure_present[i] = False
            pred_seizure_confidence[i] = 0.0
            pred_onset[i] = -1
            pred_onset_confidence[i] = 0.0
            pred_offset[i] = -1
            pred_offset_confidence[i] = 0.0
        else:
            pred_indx = pred_indx[0]
            pred_seizure_present[i] = bool(df_pred['seizure_present'][pred_indx])
            pred_seizure_confidence[i] = df_pred['seizure_confidence'][pred_indx]
            pred_onset[i] = df_pred['onset'][pred_indx]
            if np.isnan(pred_onset[i]):
                pred_onset[i] = 0
            pred_onset_confidence[i] = df_pred['onset_confidence'][pred_indx]
            if np.isnan(pred_onset_confidence[i]):
                pred_onset_confidence[i] = 0
            pred_offset[i] = df_pred['offset'][pred_indx]
            if np.isnan(pred_offset[i]):
                pred_offset[i] = 0
            pred_offset_confidence[i] = df_pred['offset_confidence'][pred_indx]
            if np.isnan(pred_offset_confidence[i]):
                pred_offset_confidence[i] = 0
    return gt_seizure_present, gt_onset, gt_offset, pred_seizure_present, pred_onset, pred_offset, pred_seizure_confidence, pred_onset_confidence, pred_offset_confidence

def compute_confusion_matrix(y_gt: np.ndarray, y_pred: np.ndarray) -> List[int]:
    """基于真实值和预测值计算混淆矩阵

    参数:
        y_gt (np.ndarray): 真实值
        y_pred (np.ndarray): 预测值

    返回:
        List[4]: [TP,FN,FP,TN]
    """
    assert len(y_gt) == len(y_pred), "数组长度不一致"
    TP = np.logical_and(y_gt, y_pred).sum()
    FN = np.logical_and(y_gt, np.logical_not(y_pred)).sum()
    FP = np.logical_and(np.logical_not(y_gt), y_pred).sum()
    TN = np.logical_and(np.logical_not(y_gt), np.logical_not(y_pred)).sum()
    return [int(TP), int(FN), int(FP), int(TN)]

def compute_basic_metrics(TP: int, FN: int, FP: int, TN: int) -> Tuple[float, float, float, float]:
    """计算分类指标：灵敏度、PPV、F1分数、准确率

    参数:
        TP (int): 真阳性
        FN (int): 假阴性
        FP (int): 假阳性
        TN (int): 真阴性

    返回:
        Tuple[float, float, float, float]: 灵敏度, PPV, F1, 准确率
    """
    if (TP + FN) == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    if (TP + FP) == 0:
        PPV = 0
    else:
        PPV = TP / (TP + FP)
    if (sensitivity + PPV) == 0:
        F1 = 0
    else:
        F1 = 2 * sensitivity * PPV / (sensitivity + PPV)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return sensitivity, PPV, F1, accuracy

def compute_detection_error(gt_seizure_present: np.ndarray, pred_seizure_present: np.ndarray, gt_time: np.ndarray, pred_time: np.ndarray, max_penalty=60) -> float:
    detection_error = (gt_seizure_present & (~pred_seizure_present)) * max_penalty \
                    + (gt_seizure_present & pred_seizure_present) * np.minimum(np.abs(pred_time - gt_time), max_penalty)
    return np.sum(detection_error) / np.sum(gt_seizure_present)

def compute_SS23_performance_metric(detection_error_onset, confusion_matrix, false_classification_pentalty=60):
    N_files = np.sum(confusion_matrix)
    [TP, FN, FP, TN] = confusion_matrix
    N_seizures = TP + FN
    metric = detection_error_onset + (FP / (FP + TN)) * false_classification_pentalty * (1 - N_seizures / N_files)
    return metric

def compute_WS23_performance_metric(gt_present: np.ndarray, pred_present: np.ndarray, gt_onset: np.ndarray, pred_onset: np.ndarray, max_delay: float = 30) -> float:
    """计算2023年冬季学期的性能指标。
        输入数组必须维度一致。

    参数:
        gt_present (np.ndarray): 真实是否有发作
        pred_present (np.ndarray): 预测是否有发作
        gt_onset (np.ndarray): 真实起始点
        pred_onset (np.ndarray): 预测起始点
        max_delay (float, optional): 允许的最大误差，默认30秒。

    返回:
        float: 性能指标
    """
    return compute_interval_scores(gt_present, pred_present, gt_onset, pred_onset, max_delay)[2]

def compute_interval_scores(gt_present: np.ndarray, pred_present: np.ndarray, gt_onset: np.ndarray, pred_onset: np.ndarray, max_delay: float = 30) -> List[float]:
    """基于预测时间是否落在真实时间区间内计算分类指标

       该指标参考一项研究，患者和医护人员对发作通知最大延迟为30秒
        https://www.sciencedirect.com/science/article/pii/S1059131116301327?pes=vor
    参数:
        gt_present (np.ndarray): 真实是否有发作
        pred_present (np.ndarray): 预测是否有发作
        gt_onset (np.ndarray): 真实起始点
        pred_onset (np.ndarray): 预测起始点
        max_delay (float, optional): 允许的最大误差，默认30秒。

    返回:
        List[float]: [I_Sensitivity, I_PPV, I_F1, I_Accuracy]
    """
    assert gt_present.shape == pred_present.shape == gt_onset.shape == pred_onset.shape, "数组长度不一致"

    in_interval = (np.abs(pred_onset - gt_onset) < max_delay)
    TP = ((gt_present & pred_present) & in_interval).sum()
    FN = (((gt_present & pred_present) & (~in_interval)) | (gt_present & (~pred_present))).sum()
    FP = ((~gt_present) & pred_present).sum()
    TN = ((~gt_present) & (~pred_present)).sum()
    return compute_basic_metrics(TP, FN, FP, TN)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store', type=str, default=r'D:\datasets\eeg\dataset_dir_original\shared_data\training_mini')
    args = parser.parse_args()
    metrics = score(args.test_dir)
    performance_metric = metrics["performance_metric_WS23"]
    F1 = metrics["F1"]
    detection_error_onset = metrics["detection_error_onset"]
    print("WKI 指标:", performance_metric, "\t F1:", F1, "\t 延迟:", detection_error_onset)
