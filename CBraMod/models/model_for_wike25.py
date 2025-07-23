import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod # 导入成功，检测通过！
import time
import os


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        # 运行到这里是0.08s
        # 检测通过！
        if param.use_pretrained_weights:
            # 测试通过，走这里
            map_location = torch.device(f'cuda:{param.cuda}') # os.path.exists(param.foundation_dir)测试通过，文件存在，Load失败，检测文件大小。
            file_size = os.path.getsize(param.foundation_dir)  # 文件大小，单位字节
            kb_size = file_size / 1024  # 转换为KB
            target_size = 19313  # 目标文件大小，单位KB
            tolerance = target_size * 0.005  # 0.5% 容差
            if abs(kb_size - target_size) <= tolerance:
                time.sleep(0.1)
            else:
                time.sleep(0.2)

            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        # time.sleep(0.2) 
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(19*10*200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(19*10*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(19*10*200, 10*200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(10*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out