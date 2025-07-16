import argparse, os, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, auc, roc_auc_score, confusion_matrix
from tqdm import tqdm

# from datasets.shu_dataset import LoadDataset
# from datasets.chb_dataset import LoadDataset
from datasets.wike25_dataset import LoadDataset
# from models.model_for_shu import Model
# from models.model_for_chb import Model
from models.model_for_wike25 import Model

def binary_entropy_from_logits(logits, eps=1e-8):
    p = torch.sigmoid(logits)
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return ent   # shape [B]

def metric_binary(model, loader, device, disable_grad=False):
    ctx = torch.no_grad() if disable_grad else torch.enable_grad()
    with ctx:
        model.eval()
        ys, ps = [], []
        for x, y in tqdm(loader, desc="Eval", mininterval=1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            prob1 = torch.sigmoid(logits).view(-1).detach().cpu()
            ys.append(y.view(-1).cpu())
            ps.append(prob1)
        y_true = torch.cat(ys).numpy().astype(int)
        y_prob = torch.cat(ps).numpy()
        y_pred = (y_prob > 0.5).astype(int)

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        return dict(bal_acc=bal_acc, pr_auc=pr_auc, roc_auc=roc_auc, cm=cm)

def smart_load(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and all('.' in k for k in obj.keys()):
        return obj
    for key in ('model', 'state_dict'):
        if key in obj:
            return obj[key]
    raise RuntimeError(f'Un-recognised checkpoint format: {list(obj.keys())}')

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--datasets_dir', default='D:\\datasets\\eeg\\dataset_processed\\CHB-MIT_seg')
    parser.add_argument('--datasets_dir', default='D:\\datasets\\eeg\\dataset_processed\\shared_data')
    parser.add_argument('--ckpt', default='./pretrained_weights/pretrained_weights.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size',   type=int, default=32,
                    help='batch size for test loader')
    parser.add_argument('--num_workers',  type=int, default=4,
                        help='dataloader worker threads')
    parser.add_argument('--use_pretrained_weights', type=bool, default=True)
    parser.add_argument('--foundation_dir',
                        default='pretrained_weights/pretrained_weights.pth')
    parser.add_argument('--classifier',
                        default='all_patch_reps',
                        choices=['all_patch_reps',
                                'all_patch_reps_twolayer',
                                'all_patch_reps_onelayer',
                                'avgpooling_patch_reps'])
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 0)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    loaders = LoadDataset(args).get_data_loader()
    test_loader = loaders['test']

    model = Model(args).to(device)
    model.load_state_dict(smart_load(args.ckpt, device), strict=False)

    # 只保留baseline推理
    base_metrics = metric_binary(model, test_loader, device, disable_grad=True)

    # print('\n=== CHB-MIT | CBraMod ===')
    print('\n=== wike25 | CBraMod ===')
    print(f'Batch Size: {args.batch_size}')
    print(f'Baseline : {base_metrics}')

if __name__ == '__main__':
    main()
