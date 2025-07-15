# 项目研讨课“医学中的人工智能竞赛”2025年夏季学期
本仓库包含了项目研讨课“医学中的人工智能竞赛”2025年夏季学期的演示代码。该示例同时定义了我们评估系统的接口。

## 入门指南

1. 克隆/复制本仓库
2. 在 github/gitlab 上创建你自己的私有仓库。之后你们可以通过该仓库提交模型作业。
3. 将我们的账号添加为协作者（Github: 名称="wettbewerbKI"，Gitlab: 可按需提供）
3. 创建一个 Python 环境。推荐使用 [Anaconda](https://www.anaconda.com/products/distribution)，并运行 `conda create -n wki-sose25 python=3.8`
4. 在该环境中安装我们使用的所有包，这些包已在 "requirements.txt" 文件中列出，安装命令如下：
```
 conda activate wki-sose25
 pip install -r requirements.txt
```
5. 下载训练数据（在 moodle 或 epilepsy-server 上有链接），解压后放在一个文件夹中（路径在我们的训练脚本中已硬编码）
5. 测试一遍所有流程，确保一切正常运行。依次执行我们的训练、预测和评分代码（并相应调整测试数据文件夹路径）：
```
python train.py
python predict_pretrained.py --test_dir ../test/
python score.py --test_dir ../test/
```

## 重要提示！

请在提交时确保我们提供的所有文件都位于仓库的顶层目录。请使用 predict_pretrained.py 脚本测试你的代码是否能正常运行。

以下文件
- predict_pretrained.py
- wettbewerb.py
- score.py

在我们的测试过程中会被还原为原始版本，因此不建议修改这些文件。在 predict.py 文件中，`predict_labels` 函数定义了我们用于评估的接口。

`predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model.json') -> Dict[str,Any]`

特别是 `model_name` 参数，你可以用它来区分不同的模型，例如通过你的文件夹结构来表示。

请在提交用于评估的 requirements.txt 文件中列出所有用到的包，并在全新环境下用 `pip install -r requirements.txt` 进行测试。你始终可以以我们提供的 "requirements.txt" 文件为基础。我们使用 Python 3.8。如果有的包只能在其他 Python 版本下运行，也可以，届时请注明所用的 Python 版本。
