# base
这个数据集少额外**中线Fpz**和**枕中线Oz**的数据。

# wike25 code部分

## [train.py]
调用了load_references，这个函数默认只读取从idx开始的100个记录。


## [wettbewerb.py]
明确说明不可以修改。如果使用12个通道的对比，需要自己写处理函数。


# CBraMod中
## [./preprocessing/preprocessing_wike25] 和 [./preprocessing/CHB-MIT/process1.py & process2.py]
1. CHB-MIT读取出来就是双极导联，而wike25给的是单极导联，是否需要做差值？
2. 我对每个电机做了归一化处理和缩放，来保证和CHB-MIT可视化效果一样的波纹。

## [./datasets/chb_dataset.py]
使用CHB-MIT数据，一共16个双电极数据，而wike25提供19条单通道，reshape需要修改为19

## [./models/model_for_chb.py]
这里也需要将16改为19


## 数据
### CHB-MIT
``` python
TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]
```

### wike25
```python
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4',
    'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]
```
