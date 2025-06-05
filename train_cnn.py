#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 1-D CNN for seizure detection + onset estimation.
Outputs:
  ├── cnn1d.pt        # 网络权重
  └── model.json      # 元数据（predict.py 会加载）
保持与 Lecture_Example 相同的读取逻辑。
"""

import os, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from tqdm import tqdm

from wettbewerb import load_references, get_3montages

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FS = 256            # 统一采样率
WIN_SEC   = 4.0            # 窗长（秒）
STEP_SEC  = 2.0            # 步长（秒）
WIN_SAMP  = int(TARGET_FS * WIN_SEC)
STEP_SAMP = int(TARGET_FS * STEP_SEC)

# ------------------------------------------------------------------ #
#                          Dataset                                   #
# ------------------------------------------------------------------ #
class EEGWindowSet(Dataset):
    """Create fixed-length windows and per-window binary labels."""
    def __init__(self, root="/home/ruicong/wki-sose25/mini_mat_wki"):
        ids, chs, data, fs, refs, labels = load_references(root)
        self.X, self.y = [], []

        for i in tqdm(range(len(ids)), desc="Building dataset"):
            # 1. montage to 3 differential channels
            _, mdata, _ = get_3montages(chs[i], data[i])
            # 2. resample
            mdata = signal.resample_poly(mdata, TARGET_FS, int(fs[i]), axis=1)
            # 3. z-score per channel
            mdata = (mdata - mdata.mean(1, keepdims=True)) / (mdata.std(1, keepdims=True) + 1e-7)

            seiz_present, onset, offset = labels[i]

            # 4. sliding windows
            n_seg = max(0, (mdata.shape[1] - WIN_SAMP) // STEP_SAMP + 1)
            for k in range(n_seg):
                s = k * STEP_SAMP
                e = s + WIN_SAMP
                self.X.append(mdata[:, s:e])

                if seiz_present:
                    t_start = s / TARGET_FS
                    t_end   = e / TARGET_FS
                    label = int((t_start <= offset) and (t_end >= onset))
                else:
                    label = 0
                self.y.append(label)

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ------------------------------------------------------------------ #
#                         Model                                      #
# ------------------------------------------------------------------ #
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)           # (B,256,1)
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

# ------------------------------------------------------------------ #
#                         Training                                   #
# ------------------------------------------------------------------ #
def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train():
    set_seed()
    ds = EEGWindowSet()
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)

    model = CNN1D().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    EPOCHS, best_f1 = 50, 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        running_loss, tp, fp, fn = 0.0, 0, 0, 0

        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pred = logits.argmax(1)
            tp += ((pred == 1) & (yb == 1)).sum().item()
            fp += ((pred == 1) & (yb == 0)).sum().item()
            fn += ((pred == 0) & (yb == 1)).sum().item()

        sens = tp / (tp + fn + 1e-9)
        ppv  = tp / (tp + fp + 1e-9)
        f1   = 2 * sens * ppv / (sens + ppv + 1e-9)

        print(f"Epoch {ep:02d}  loss: {running_loss/len(ds):.4f}  F1: {f1:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "cnn1d.pt")
    with open("model.json", "w") as f:
        json.dump({
            "cnn_weight_path": "cnn1d.pt",
            "prob_th": 0.6,
            "min_len": 2,            # 连续窗口数阈值
            "win_sec": WIN_SEC,
            "step_sec": STEP_SEC,
            "fs": TARGET_FS,
            "std_thresh": 0          # 占位，兼容旧代码
        }, f)
    print("模型已保存: cnn1d.pt / model.json")

if __name__ == "__main__":
    train()
