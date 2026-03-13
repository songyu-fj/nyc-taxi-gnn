"""
train_final.py
训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('..')
from models.multi_graph_gcn_final import RobustMultiGraphGCN
from create_dataloader_final import create_industrial_dataloaders

# ==================== 配置 ====================
BATCH_SIZE = 64
WINDOW_SIZE = 9
HORIZON = 3
HIDDEN_DIM = 64
NUM_BLOCKS = 2
EPOCHS = 50
LR = 0.001
ACCUMULATION_STEPS = 2
SAVE_DIR = '../results/paper_experiment/'

# ==================== 训练器 ====================
class SafeMetrics:
    @staticmethod
    def mae(pred, target): return torch.mean(torch.abs(pred - target))
    @staticmethod
    def rmse(pred, target): return torch.sqrt(torch.mean((pred - target) ** 2))
    @staticmethod
    def mape(pred, target, eps=1e-4):
        mask = torch.abs(target) > eps
        if mask.sum() == 0:
            return torch.tensor(float('nan'))
        return torch.mean(torch.abs((pred[mask] - target[mask]) / target[mask])) * 100

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4, save_path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save({'model_state_dict': model.state_dict(), 'best_score': self.best_score}, self.save_path)

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, norm_params, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.norm_params = norm_params
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.save_dir = SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': []}

    def denormalize(self, tensor):
        return tensor * self.norm_params['std'] + self.norm_params['mean']

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        self.optimizer.zero_grad()
        for i, (x, y, adj_s, adj_f, adj_p) in enumerate(pbar, 1):
            x, y = x.to(self.device), y.to(self.device)
            adj_s, adj_f, adj_p = adj_s.to(self.device), adj_f.to(self.device), adj_p.to(self.device)

            pred = self.model(x, adj_s, adj_f, adj_p)
            loss = self.criterion(pred, y) / ACCUMULATION_STEPS
            loss.backward()

            if i % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({'loss': f'{loss.item()*ACCUMULATION_STEPS:.4f}'})
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y, adj_s, adj_f, adj_p in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                adj_s, adj_f, adj_p = adj_s.to(self.device), adj_f.to(self.device), adj_p.to(self.device)
                pred = self.model(x, adj_s, adj_f, adj_p)
                all_pred.append(pred.cpu())
                all_true.append(y.cpu())
        pred = torch.cat(all_pred)
        true = torch.cat(all_true)
        loss = self.criterion(pred, true).item()
        pred_real = self.denormalize(pred)
        true_real = self.denormalize(true)
        mae = SafeMetrics.mae(pred_real, true_real).item()
        rmse = SafeMetrics.rmse(pred_real, true_real).item()
        return loss, mae, rmse

    def test(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y, adj_s, adj_f, adj_p in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                adj_s, adj_f, adj_p = adj_s.to(self.device), adj_f.to(self.device), adj_p.to(self.device)
                pred = self.model(x, adj_s, adj_f, adj_p)
                all_pred.append(pred.cpu())
                all_true.append(y.cpu())
        pred = torch.cat(all_pred)
        true = torch.cat(all_true)
        pred_real = self.denormalize(pred)
        true_real = self.denormalize(true)
        mae = SafeMetrics.mae(pred_real, true_real).item()
        rmse = SafeMetrics.rmse(pred_real, true_real).item()
        mape = SafeMetrics.mape(pred_real, true_real).item()
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    def run(self):
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        early_stopper = EarlyStopping(patience=15, save_path=best_model_path)

        for epoch in range(1, EPOCHS+1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_mae, val_rmse = self.validate()
            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)

            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f}")

            early_stopper(val_loss, self.model)
            if early_stopper.early_stop:
                print("早停触发")
                break

        test_metrics = self.test(best_model_path)
        print("\n🏆 测试集结果:")
        print(f"   MAE : {test_metrics['MAE']:.4f}")
        print(f"   RMSE: {test_metrics['RMSE']:.4f}")
        print(f"   MAPE: {test_metrics['MAPE']:.2f}%")

        np.save(os.path.join(self.save_dir, 'history.npy'), self.history)
        return test_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader, norm_params = create_industrial_dataloaders(
        batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, horizon=HORIZON
    )

    sample_x = next(iter(train_loader))[0]
    num_nodes = sample_x.shape[1]

    model = RobustMultiGraphGCN(
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        block_hidden=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS
    )

    trainer = Trainer(model, train_loader, val_loader, test_loader, norm_params, device)
    trainer.run()

if __name__ == '__main__':
    main()