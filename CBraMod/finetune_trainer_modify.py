import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F

from finetune_evaluator import Evaluator


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        
        # Enable torch.compile for faster training (PyTorch 2.0+) - disabled by default due to Triton dependency
        if hasattr(torch, 'compile') and getattr(params, 'use_compile', False):
            try:
                self.model = torch.compile(self.model)
                print("torch.compile enabled successfully")
            except Exception as e:
                print(f"torch.compile failed, falling back to eager mode: {e}")
        
        # Mixed precision training
        self.use_amp = getattr(params, 'use_amp', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Gradient accumulation
        # 由于数据已经预批处理，可以减少梯度累积步数
        self.accumulation_steps = getattr(params, 'accumulation_steps', 1)
        
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB', "wike25"]:
            # 对于癫痫检测，使用自定义损失函数
            if self.params.downstream_dataset == "wike25":
                self.pos_weight = self._calculate_class_weights()
                # 基础BCE损失，但会在train_for_binaryclass中使用自定义损失
                self.criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight).cuda()
                print(f"使用自定义癫痫检测损失，正样本权重: {self.pos_weight.item():.3f}")
            else:
                self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length // self.accumulation_steps, eta_min=1e-6
        )
        print(self.model)

    def _calculate_class_weights(self):
        """计算类别权重以处理不平衡数据"""
        print("正在计算类别权重...")
        total_samples = 0
        positive_samples = 0
        
        for batch_data in self.data_loader['train']:
            _, labels = batch_data
            total_samples += len(labels)
            positive_samples += torch.sum(labels).item()
        
        negative_samples = total_samples - positive_samples
        
        print(f"训练集统计:")
        print(f"  总样本数: {total_samples}")
        print(f"  负样本数: {negative_samples} ({negative_samples/total_samples*100:.1f}%)")
        print(f"  正样本数: {positive_samples} ({positive_samples/total_samples*100:.1f}%)")
        print(f"  不平衡比例: {negative_samples/positive_samples:.1f}:1")
        
        # 计算正样本权重 = 负样本数 / 正样本数，并确保在GPU上
        pos_weight = torch.tensor([negative_samples / positive_samples], dtype=torch.float32).cuda()
        return pos_weight

    def _calculate_seizure_loss(self, pred, target):
        """
        平衡的癫痫检测损失函数
        使用加权BCE + 适度的正样本激励
        """
        # 确保pos_weight在正确的设备上
        pos_weight = self.pos_weight.to(pred.device)
        
        # 主损失：加权BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight, reduction='mean'
        )
        
        # 适度的正样本置信度要求（降低要求从0.7到0.6）
        sigmoid_pred = torch.sigmoid(pred)
        positive_confidence_loss = target * torch.clamp(0.6 - sigmoid_pred, min=0) * 1.0  # 降低权重
        
        # 组合损失：主要是加权BCE + 轻微的正样本置信度要求
        combined_loss = bce_loss + positive_confidence_loss.mean()
        
        return combined_loss

    def _calculate_balanced_accuracy(self, pred, target):
        """计算平衡准确率（固定阈值0.5）"""
        sigmoid_pred = torch.sigmoid(pred)
        pred_binary = (sigmoid_pred > 0.5).float()
        
        # 计算每个类别的准确率
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        
        if pos_mask.sum() > 0:
            pos_acc = ((pred_binary == target) & pos_mask).float().sum() / pos_mask.sum()
        else:
            pos_acc = torch.tensor(0.0)
            
        if neg_mask.sum() > 0:
            neg_acc = ((pred_binary == target) & neg_mask).float().sum() / neg_mask.sum()
        else:
            neg_acc = torch.tensor(0.0)
            
        balanced_acc = (pos_acc + neg_acc) / 2.0
        return balanced_acc.item(), pos_acc.item(), neg_acc.item()

    def _find_optimal_threshold(self, model):
        """在验证集上寻找最优分类阈值"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in self.data_loader['val']:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                pred = model(x)
                sigmoid_pred = torch.sigmoid(pred)
                
                all_preds.extend(sigmoid_pred.cpu().numpy().tolist())
                all_targets.extend(y.cpu().numpy().tolist())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 尝试不同阈值，找到最佳F1分数
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            pred_binary = (all_preds > threshold).astype(int)
            
            # 计算F1分数
            tp = np.sum((all_targets == 1) & (pred_binary == 1))
            fp = np.sum((all_targets == 0) & (pred_binary == 1))
            fn = np.sum((all_targets == 1) & (pred_binary == 0))
            
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
        
        return best_threshold, best_f1

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            self.optimizer.zero_grad()
            
            pbar = tqdm(self.data_loader['train'], mininterval=1)
            for i, (x, y) in enumerate(pbar):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                
                with autocast('cuda', enabled=self.use_amp):
                    pred = self.model(x)
                    if self.params.downstream_dataset == 'ISRUC':
                        loss = self.criterion(pred.transpose(1, 2), y)
                    else:
                        loss = self.criterion(pred, y)
                    loss = loss / self.accumulation_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                losses.append(loss.data.cpu().numpy() * self.accumulation_steps)
                
                # Update progress bar with current metrics
                current_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
                pbar.set_description(f"Epoch {epoch+1}/{self.params.epochs} - Loss: {current_loss:.5f}")
                
                if (i + 1) % self.accumulation_steps == 0:
                    if self.params.clip_value > 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer_scheduler.step()
                    self.optimizer.zero_grad()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        f1_best = 0
        balanced_acc_best = 0
        
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            balanced_accs = []
            pos_accs = []
            neg_accs = []
            self.optimizer.zero_grad()
            
            pbar = tqdm(self.data_loader['train'], mininterval=1)
            for i, (x, y) in enumerate(pbar):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                
                with autocast('cuda', enabled=self.use_amp):
                    pred = self.model(x)
                    
                    # 使用平衡的自定义损失函数
                    if self.params.downstream_dataset == "wike25":
                        loss = self._calculate_seizure_loss(pred, y)
                    else:
                        loss = self.criterion(pred, y)
                    
                    loss = loss / self.accumulation_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                losses.append(loss.data.cpu().numpy() * self.accumulation_steps)
                
                # 计算训练时的平衡准确率（固定阈值0.5）
                with torch.no_grad():
                    bal_acc, pos_acc, neg_acc = self._calculate_balanced_accuracy(pred, y)
                    balanced_accs.append(bal_acc)
                    pos_accs.append(pos_acc)
                    neg_accs.append(neg_acc)
                
                # Update progress bar with current metrics
                current_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
                current_bal_acc = np.mean(balanced_accs[-50:]) if len(balanced_accs) >= 50 else np.mean(balanced_accs)
                pbar.set_description(f"Epoch {epoch+1}/{self.params.epochs} - Loss: {current_loss:.5f} - BalAcc: {current_bal_acc:.3f}")
                
                if (i + 1) % self.accumulation_steps == 0:
                    if self.params.clip_value > 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer_scheduler.step()
                    self.optimizer.zero_grad()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                # 使用原始evaluator获取指标（固定阈值0.5）
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                
                # 计算指标（固定阈值0.5）
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                balanced_accuracy = (recall + specificity) / 2.0
                
                # 计算训练集平均指标
                train_bal_acc = np.mean(balanced_accs)
                train_pos_acc = np.mean(pos_accs)
                train_neg_acc = np.mean(neg_accs)
                
                print(
                    "Epoch {} : Training Loss: {:.5f}, val_acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, f1: {:.5f}, bal_acc: {:.5f}, LR: {:.5f}, Time: {:.2f}min".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        f1,
                        balanced_accuracy,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(f"训练集: 平衡准确率={train_bal_acc:.5f}, 正类准确率={train_pos_acc:.5f}, 负类准确率={train_neg_acc:.5f}")
                print(f"验证集(阈值=0.5): TN={tn}, FP={fp}, FN={fn}, TP={tp}")
                print(f"验证集: Precision={precision:.5f}, Recall={recall:.5f}, Specificity={specificity:.5f}")
                
                # 每个epoch都保存一次模型
                if not os.path.isdir(self.params.model_dir):
                    os.makedirs(self.params.model_dir)
                model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}_bal_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(
                    epoch + 1, f1, balanced_accuracy, pr_auc, roc_auc)
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)

                # 使用综合指标选择最佳模型：平衡准确率优先，兼顾F1和ROC-AUC
                current_composite_score = (
                    0.4 * balanced_accuracy +  # 平衡准确率权重40%
                    0.3 * f1 +                 # F1分数权重30%  
                    0.3 * roc_auc              # ROC-AUC权重30%
                )
                
                best_composite_score = (
                    0.4 * balanced_acc_best +
                    0.3 * f1_best +
                    0.3 * roc_auc_best
                )
                
                if current_composite_score > best_composite_score:
                    print("综合指标提升！保存最佳模型...")
                    print("Val Evaluation: f1: {:.5f}, bal_acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, composite: {:.5f}".format(
                        f1, balanced_accuracy, pr_auc, roc_auc, current_composite_score
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    f1_best = f1
                    balanced_acc_best = balanced_accuracy
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    
        # 测试最佳模型
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            print("使用固定阈值: 0.5")
            
            # 使用原始evaluator获取测试集指标
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            
            # 计算测试集完整指标
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            balanced_accuracy = (recall + specificity) / 2.0
            
            # 计算综合得分
            final_composite_score = (
                0.4 * balanced_accuracy +
                0.3 * f1 +
                0.3 * roc_auc
            )
            
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, f1: {:.5f}, bal_acc: {:.5f}, composite: {:.5f}".format(
                    acc, pr_auc, roc_auc, f1, balanced_accuracy, final_composite_score
                )
            )
            print(f"测试集混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            print(f"测试集 Precision: {precision:.5f}, Recall: {recall:.5f}, Specificity: {specificity:.5f}")
            print(f"癫痫检出率(Sensitivity/Recall): {recall:.5f}")
            print(f"健康识别率(Specificity): {specificity:.5f}")
            
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/BEST_f1_{:.5f}_bal_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(
                f1, balanced_accuracy, pr_auc, roc_auc)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'threshold': 0.5,
                'test_metrics': {
                    'accuracy': acc,
                    'f1_score': f1,
                    'balanced_accuracy': balanced_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'composite_score': final_composite_score
                }
            }, model_path)
            print("best model save in " + model_path)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            self.optimizer.zero_grad()
            
            pbar = tqdm(self.data_loader['train'], mininterval=1)
            for i, (x, y) in enumerate(pbar):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                
                with autocast('cuda', enabled=self.use_amp):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    loss = loss / self.accumulation_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                losses.append(loss.data.cpu().numpy() * self.accumulation_steps)
                
                # Update progress bar with current metrics
                current_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
                pbar.set_description(f"Epoch {epoch+1}/{self.params.epochs} - Loss: {current_loss:.5f}")
                
                if (i + 1) % self.accumulation_steps == 0:
                    if self.params.clip_value > 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer_scheduler.step()
                    self.optimizer.zero_grad()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)