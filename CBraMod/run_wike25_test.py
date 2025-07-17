import argparse
import torch
import numpy as np
from datasets import wike25_dataset
from models import model_for_wike25
from finetune_evaluator import Evaluator


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test_model():
    # 设置参数，参考 finetune_main.py 中的默认值
    parser = argparse.ArgumentParser(description='Test wike25 model')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for testing')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps', help='classifier type')
    parser.add_argument('--downstream_dataset', type=str, default='wike25', help='dataset name')
    parser.add_argument('--datasets_dir', type=str,
                        default=r'D:\datasets\eeg\dataset_processed\shared_data',
                        help='datasets directory')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--use_pretrained_weights', type=bool, default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)

    # 加载数据集
    print("Loading dataset...")
    load_dataset = wike25_dataset.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()

    # 创建模型
    print("Creating model...")
    model = model_for_wike25.Model(params)
    model = model.cuda()

    # 加载预训练权重
    model_path = "model_weights/wike25/epoch5_acc_0.79077_pr_0.74238_roc_0.87214.pth"
    print(f"Loading model weights from {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=f'cuda:{params.cuda}')
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # 创建评估器
    print("Creating evaluator...")
    test_eval = Evaluator(params, data_loader['test'])

    # 进行测试
    print("Starting evaluation...")
    model.eval()

    with torch.no_grad():
        # 由于 wike25 是二分类任务，使用 get_metrics_for_binaryclass
        acc, pr_auc, roc_auc, cm = test_eval.get_metrics_for_binaryclass(model)

        print("=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(f"Accuracy: {acc:.5f}")
        print(f"PR AUC: {pr_auc:.5f}")
        print(f"ROC AUC: {roc_auc:.5f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("=" * 50)


if __name__ == '__main__':
    test_model()