# -*- coding: utf-8 -*-
"""
推理脚本：载入 model.json + cnn1d.pt
与主办方接口保持一致。
"""

from typing import List, Dict, Any
import json, torch, torch.nn as nn, numpy as np
from scipy import signal
from wettbewerb import get_3montages

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- 网络结构需匹配训练 --------
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, 2, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256, 2)
    def forward(self, x):
        x = self.features(x).squeeze(-1)
        return self.classifier(x)

# -------- 接口函数 --------
def predict_labels(channels: List[str], data: np.ndarray,
                   fs: float, reference_system: str,
                   model_name: str = "model.json") -> Dict[str, Any]:

    # 1. 读取元数据 & 模型
    with open(model_name, "r") as f:
        params = json.load(f)

    model = CNN1D().to(DEVICE).eval()
    state_dict = torch.load(params["cnn_weight_path"], map_location=DEVICE)
    model.load_state_dict(state_dict)

    prob_th  = params["prob_th"]
    min_len  = params["min_len"]
    win_sec  = params["win_sec"]
    step_sec = params["step_sec"]
    tgt_fs   = params["fs"]

    # 2. 预处理
    _, mdata, _ = get_6montages(channels, data)
    mdata = signal.resample_poly(mdata, tgt_fs, int(fs), axis=1)
    mdata = (mdata - mdata.mean(1, keepdims=True)) / (mdata.std(1, keepdims=True) + 1e-7)

    win_samp  = int(win_sec * tgt_fs)
    step_samp = int(step_sec * tgt_fs)
    n_seg = max(0, (mdata.shape[1] - win_samp) // step_samp + 1)

    if n_seg == 0:   # 录音太短
        return {"seizure_present": False,
                "seizure_confidence": 0.,
                "onset": -1,
                "onset_confidence": 0.,
                "offset": -1,
                "offset_confidence": 0.}

    segs = np.stack([mdata[:, s:s+win_samp] for s in range(0, n_seg*step_samp, step_samp)])
    with torch.no_grad():
        logits = model(torch.tensor(segs, dtype=torch.float32).to(DEVICE))
        probs  = torch.softmax(logits, 1)[:, 1].cpu().numpy()

    # 3. 平滑 + 阈值
    smooth = np.convolve(probs, np.ones(3)/3, mode="same")
    mask = smooth > prob_th

    # 连通域阈值
    runs, current = [], []
    for i, m in enumerate(mask):
        if m: current.append(i)
        elif current:
            if len(current) >= min_len: runs.append(current)
            current = []
    if current and len(current) >= min_len: runs.append(current)

    if not runs:     # 无癫痫
        return {"seizure_present": False,
                "seizure_confidence": float(smooth.max()),
                "onset": -1, "onset_confidence": 0.,
                "offset": -1, "offset_confidence": 0.}

    # 取最长 run
    run = max(runs, key=len)
    onset_sec  = run[0] * step_sec
    offset_sec = run[-1] * step_sec + win_sec

    conf = float(smooth[run].max())
    return {"seizure_present": True,
            "seizure_confidence": conf,
            "onset": onset_sec,
            "onset_confidence": conf,
            "offset": offset_sec,
            "offset_confidence": conf}
