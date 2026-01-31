#AutoEncoder訓練系統
"""
CLI Mode:

Added support for Command Line Interface (CLI) execution using argparse.
Available arguments:
--file <path>: Path to input data (Excel/CSV).
--simulate: Use simulated data.
--aug_factor <int>: Data augmentation factor (default 6).
--include_original: Helper flag to include original data in the training set during augmentation.
--epochs <int>: Max training epochs (default 500).
--patience <int>: Early stopping patience (default 50).
--save_dir <path>: Directory to save results.
Example: python spectra_ae_system.py --file data.xlsx --aug_factor 4 --include_original


主要功能
1. 5種模型架構

SimpleAE: 簡單2層編碼-解碼結構
DeepAE: 深層網絡,帶BatchNorm和Dropout
ResidualAE: 殘差連接,提升梯度流動
WideAE: 寬層架構,更多神經元
BottleneckAE: 瓶頸結構,強制特徵壓縮

2. 4種數據縮放器

StandardScaler (標準化)
MinMaxScaler (歸一化)
RobustScaler (抗離群值)
NoScaler (不縮放)

3. 6種數據增強方法

無增強
高斯噪聲
強度縮放
基線偏移
Mixup混合
組合增強

4. 自動化流程

自動測試所有組合 (5×4×6 = 120種配置)
Early stopping防止過擬合
學習率自適應調整
自動保存前3名模型
生成詳細benchmark報告

5. 評估指標

MSE (均方誤差)
MAE (平均絕對誤差)
RMSE (均方根誤差)
R² (決定係數)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import json
import os
import pickle
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 設定中文字型 (如果您在Windows環境)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 數據集定義 ====================
class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 數據增強方法 ====================
class SpectraAugmentation:
    @staticmethod
    def no_augmentation(X, y):
        return X, y
    
    @staticmethod
    def gaussian_noise(X, y, noise_level=0.01):
        """添加高斯噪聲"""
        X_aug = X + np.random.normal(0, noise_level, X.shape)
        y_aug = y + np.random.normal(0, noise_level, y.shape)
        return X_aug, y_aug
    
    @staticmethod
    def intensity_scaling(X, y, scale_range=(0.95, 1.05)):
        """強度縮放"""
        scale = np.random.uniform(scale_range[0], scale_range[1], (X.shape[0], 1))
        X_aug = X * scale
        y_aug = y * scale
        return X_aug, y_aug
    
    @staticmethod
    def baseline_shift(X, y, shift_range=(-0.02, 0.02)):
        """基線偏移"""
        shift = np.random.uniform(shift_range[0], shift_range[1], (X.shape[0], 1))
        X_aug = X + shift
        y_aug = y + shift
        return X_aug, y_aug
    
    @staticmethod
    def mixup(X, y, alpha=0.2):
        """Mixup數據增強"""
        if len(X) < 2:
            return X, y
        indices = np.random.permutation(len(X))
        lam = np.random.beta(alpha, alpha, len(X))
        lam = lam.reshape(-1, 1)
        X_aug = lam * X + (1 - lam) * X[indices]
        y_aug = lam * y + (1 - lam) * y[indices]
        return X_aug, y_aug
    
    @staticmethod
    def combined_augmentation(X, y):
        """組合多種增強方法"""
        X_aug, y_aug = X.copy(), y.copy()
        # 50%機率應用噪聲
        if np.random.rand() > 0.5:
            X_aug, y_aug = SpectraAugmentation.gaussian_noise(X_aug, y_aug, 0.005)
        # 50%機率應用強度縮放
        if np.random.rand() > 0.5:
            X_aug, y_aug = SpectraAugmentation.intensity_scaling(X_aug, y_aug)
        # 30%機率應用基線偏移
        if np.random.rand() > 0.7:
            X_aug, y_aug = SpectraAugmentation.baseline_shift(X_aug, y_aug)
        return X_aug, y_aug


# ==================== AutoEncoder模型架構 ====================
class SimpleAE(nn.Module):
    """簡單的AutoEncoder"""
    def __init__(self, input_dim=18, output_dim=26):
        super(SimpleAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim*4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim*4, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, output_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAE(nn.Module):
    """深層AutoEncoder"""
    def __init__(self, input_dim=18, output_dim=26):
        super(DeepAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.BatchNorm1d(input_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim*2, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim*4, input_dim*8),
            nn.BatchNorm1d(input_dim*8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim*8, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim*4, input_dim*2),
            nn.BatchNorm1d(input_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim*2, output_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ResidualBlock(nn.Module):
    """殘差塊"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.block(x))


class ResidualAE(nn.Module):
    """帶殘差連接的AutoEncoder"""
    def __init__(self, input_dim=18, output_dim=26):
        super(ResidualAE, self).__init__()
        self.input_proj = nn.Linear(input_dim, input_dim*4)
        self.res_blocks = nn.Sequential(
            ResidualBlock(input_dim*4),
            ResidualBlock(input_dim*4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim*4, input_dim*2),
            nn.BatchNorm1d(input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*8, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.ReLU(),
            nn.Linear(input_dim*4, output_dim)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x


class WideAE(nn.Module):
    """寬層AutoEncoder"""
    def __init__(self, input_dim=18, output_dim=26):
        super(WideAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*8),
            nn.BatchNorm1d(input_dim*8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim*8, input_dim*16),
            nn.BatchNorm1d(input_dim*16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim*16, input_dim*8),
            nn.BatchNorm1d(input_dim*8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim*8, output_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class BottleneckAE(nn.Module):
    """瓶頸型AutoEncoder"""
    def __init__(self, input_dim=18, output_dim=26):
        super(BottleneckAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim//2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, output_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==================== 訓練器 ====================

# ==================== 訓練器 ====================
class SpectraAETrainer:
    def __init__(self, save_dir='spectra_ae_results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        # 定義所有配置
        self.models = {
            'SimpleAE': SimpleAE,
            'DeepAE': DeepAE,
            'ResidualAE': ResidualAE,
            'WideAE': WideAE,
            'BottleneckAE': BottleneckAE
        }
        
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'NoScaler': None
        }
        
        self.augmentations = {
            'None': SpectraAugmentation.no_augmentation,
            'GaussianNoise': SpectraAugmentation.gaussian_noise,
            'IntensityScaling': SpectraAugmentation.intensity_scaling,
            'BaselineShift': SpectraAugmentation.baseline_shift,
            'Mixup': SpectraAugmentation.mixup,
            'Combined': SpectraAugmentation.combined_augmentation
        }
        
        self.results = []

    def load_data_from_file(self, file_path: str, input_cols: Optional[List[str]] = None, target_cols: Optional[List[str]] = None):
        """從Excel或CSV讀取並處理數據"""
        print(f"正在讀取檔案: {file_path}")
        
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("不支援的檔案格式，請使用 .xlsx, .xls 或 .csv")
            
        print(f"原始數據形狀: {df.shape}")
        
        # 自動解析列名 (如果未提供)
        if input_cols is None or target_cols is None:
            print("正在自動解析輸入/輸出列...")
            # 假設: 列名包含 'Input' 或 'X' 為輸入, 'Target' 或 'Y' 或 'Output' 為輸出
            # 或者簡單地: 數字列的一半為輸入, 一半為輸出?? 
            # 這裡採用更通用的做法: 嘗試尋找關鍵字, 否則使用類似 pytorch_regression_system 的邏輯 (後1/3為輸出)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                raise ValueError("未找到數值列!")
                
            # 簡單的啟發式規則: 有關鍵字用關鍵字, 否則切分
            input_candidates = [c for c in numeric_cols if any(k in c.upper() for k in ['IN', 'X', 'SOURCE', 'RAW'])]
            target_candidates = [c for c in numeric_cols if any(k in c.upper() for k in ['OUT', 'Y', 'TARGET', 'REF'])]
            
            if input_candidates and target_candidates:
                input_cols = input_candidates
                target_cols = target_candidates
            else:
                # 策略 1: 尋找全空列 (Blank Column) 作為分隔符
                # 這在 Excel 中常見，使用者會留一空行分隔 Input 和 Target
                split_index = -1
                for i, col in enumerate(numeric_cols):
                    # 檢查該列是否全為 NaN
                    if df[col].isna().all():
                        split_index = i
                        print(f"檢測到空白列 '{col}' (Index {i})，將以此作為分隔")
                        break
                    # 或者檢查列名是否暗示它是空的 (例如 "Unnamed" 且數據大部分是空)
                    # 這裡嚴格一點，要求全空，避免誤判
                
                if split_index != -1:
                    input_cols = numeric_cols[:split_index]
                    # 跳過分隔列本身
                    target_cols = numeric_cols[split_index+1:]
                else:
                    # 策略 2: 默認機制 (前 2/3 是輸入, 後 1/3 是輸出)
                    n_cols = len(numeric_cols)
                    split_idx = int(n_cols * 0.6) # 稍微多一點給輸入
                    input_cols = numeric_cols[:split_idx]
                    target_cols = numeric_cols[split_idx:]
                    print(f"未檢測到明確標籤或空白分隔列, 使用默認分割: 前 {len(input_cols)} 列為輸入, 後 {len(target_cols)} 列為輸出")
        
        print(f"輸入列: {len(input_cols)} (例如: {input_cols[:3]}...)")
        print(f"輸出列: {len(target_cols)} (例如: {target_cols[:3]}...)")
        
        X = df[input_cols].values
        y = df[target_cols].values
        
        # 移除含有 NaN 的行
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"有效數據: {len(X)} 筆")
        return X, y, input_cols, target_cols
    
    def prepare_data(self, X_18, y_26, scaler_name, aug_name, aug_factor=6, aug_include_ori=False, test_size=0.15):
        """準備數據"""
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X_18, y_26, test_size=test_size, random_state=42
        )
        
        # 應用縮放器
        if scaler_name != 'NoScaler':
            scaler_X = self.scalers[scaler_name].__class__()
            scaler_y = self.scalers[scaler_name].__class__()
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)
            y_train = scaler_y.fit_transform(y_train)
            y_test = scaler_y.transform(y_test)
        else:
            scaler_X = None
            scaler_y = None
        
        # 應用數據增強
        if aug_name != 'None':
            # 複製原始數據以進行增強
            X_original = X_train.copy()
            y_original = y_train.copy()
            
            # 使用列表收集所有增強數據
            if aug_include_ori:
                all_X = [X_train]
                all_y = [y_train]
            else:
                all_X = []
                all_y = []
            
            for _ in range(aug_factor):
                X_aug, y_aug = self.augmentations[aug_name](X_original, y_original)
                all_X.append(X_aug)
                all_y.append(y_aug)
            
            X_train = np.vstack(all_X)
            y_train = np.vstack(all_y)
        
        return X_train, X_test, y_train, y_test, scaler_X, scaler_y
    
    def train_model(self, model, train_loader, val_loader, epochs=500, lr=0.001, patience=50):
        """訓練模型"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        actual_epochs = 0
        
        for epoch in range(epochs):
            actual_epochs = epoch + 1
            # 訓練階段
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 驗證階段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_loss, actual_epochs
    
    def evaluate_model(self, model, test_loader):
        """評估模型"""
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        mse = total_loss / len(test_loader)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # 計算額外指標
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((all_targets - all_preds)**2) / np.sum((all_targets - np.mean(all_targets))**2)
        
        # MAPE: Mean Absolute Percentage Error
        # 注意: 如果真實值為0，MAPE未定義。這裡加入一個極小值防止除以零
        epsilon = 1e-8
        mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + epsilon))) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def run_experiment(self, X_18, y_26, model_name, scaler_name, aug_name, aug_factor=6, aug_include_ori=False, epochs=500, patience=50):
        """運行單個實驗"""
        print(f"\n測試: Model={model_name}, Scaler={scaler_name}, Aug={aug_name}, Factor={aug_factor}, IncludeOri={aug_include_ori}")
        
        # 準備數據
        X_train, X_test, y_train, y_test, scaler_X, scaler_y = self.prepare_data(
            X_18, y_26, scaler_name, aug_name, aug_factor=aug_factor, aug_include_ori=aug_include_ori
        )
        
        # 創建數據加載器
        train_dataset = SpectraDataset(X_train, y_train)
        test_dataset = SpectraDataset(X_test, y_test)
        
        # 確保 batch_size 至少為 2，避免 BatchNorm 在 batch_size=1 時報錯
        batch_size = max(2, min(32, len(X_train) // 4))
        
        # drop_last=True: 如果最後一個 batch 大小為 1，則捨棄，防止 BatchNorm 報錯
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 獲取維度
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        
        # 創建並訓練模型 (傳入動態維度)
        model = self.models[model_name](input_dim=input_dim, output_dim=output_dim).to(self.device)
        # 傳入用戶設定的 epochs 和 patience
        val_loss, actual_epochs = self.train_model(model, train_loader, test_loader, epochs=epochs, patience=patience)
        
        # 評估模型
        metrics = self.evaluate_model(model, test_loader)
        
        result = {
            'model': model_name,
            'scaler': scaler_name,
            'augmentation': aug_name,
            'aug_factor': aug_factor,
            'aug_include_ori': aug_include_ori, # 保存設定
            'val_loss': val_loss,
            'actual_epochs': actual_epochs, # 記錄實際訓練輪數
            **metrics,
            'model_state': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
        
        print(f"  MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, MAPE: {metrics['mape']:.2f}%, R²: {metrics['r2']:.4f}, Epochs: {actual_epochs}")
        
        return result
    
    def train_all_combinations(self, X_18, y_26, aug_factor=6, aug_include_ori=False, epochs=500, patience=50):
        """訓練所有組合"""
        print("=" * 70)
        print("開始訓練所有模型組合...")
        print(f"數據集大小: {len(X_18)} 筆")
        print(f"輸入維度: {X_18.shape[1]}, 輸出維度: {y_26.shape[1]}")
        print(f"增強倍數: {aug_factor} (總數 = {'包含' if aug_include_ori else '不含'}原始 + {aug_factor}x增強)")
        print(f"訓練設定: Max Epochs={epochs}, Patience={patience}")
        print("=" * 70)
        
        total_combinations = len(self.models) * len(self.scalers) * len(self.augmentations)
        current = 0
        
        for model_name in self.models.keys():
            for scaler_name in self.scalers.keys():
                for aug_name in self.augmentations.keys():
                    current += 1
                    print(f"\n[{current}/{total_combinations}]", end=" ")
                    
                    try:
                        result = self.run_experiment(
                            X_18, y_26, model_name, scaler_name, aug_name, 
                            aug_factor=aug_factor, aug_include_ori=aug_include_ori, 
                            epochs=epochs, patience=patience
                        )
                        self.results.append(result)
                    except Exception as e:
                        print(f"  錯誤: {str(e)}")
                        continue
        
        # 按 MAPE 排序 (越小越好)
        # 注意: 確保 MAPE 存在，如果失敗可能為 None/0
        self.results.sort(key=lambda x: x.get('mape', float('inf')))
        
        # 保存前3名模型
        self.save_top_models(X_18, y_26, self.results, aug_factor=aug_factor, aug_include_ori=aug_include_ori, top_k=3)
        
        # 生成報告
        self.generate_report()
        self.generate_pdf_report()
    
    def save_top_models(self, X_sample, y_sample, results, aug_factor=6, aug_include_ori=False, top_k=3):
        """保存前K個最佳模型"""
        print(f"\n保存前{top_k}個最佳模型 (依 MAPE 排序)...")
        # 注意: 這裡 results 參數應該已經排好序了
        
        for i, result in enumerate(results[:top_k], 1):
            model_dir = os.path.join(self.save_dir, f'top_{i}_model')
            os.makedirs(model_dir, exist_ok=True)
            
            # 1. 保存模型權重
            torch.save(
                result['model_state'],
                os.path.join(model_dir, 'model.pth')
            )
            
            # 2. 保存Scaler
            if result['scaler_X'] is not None:
                joblib.dump(result['scaler_X'], os.path.join(model_dir, 'scaler_X.pkl'))
                joblib.dump(result['scaler_y'], os.path.join(model_dir, 'scaler_y.pkl'))
            
            # 3. 保存配置
            config = {
                'rank': i,
                'model_class': result['model'],
                'scaler_type': result['scaler'],
                'augmentation': result['augmentation'],
                'aug_factor': result.get('aug_factor', aug_factor),
                'aug_include_ori': result.get('aug_include_ori', aug_include_ori), # 新增
                'actual_epochs': result.get('actual_epochs', 0),
                'input_dim': X_sample.shape[1],
                'output_dim': y_sample.shape[1],
                'metrics': {
                    'mse': float(result['mse']),
                    'mae': float(result['mae']),
                    'mape': float(result.get('mape', 0)),
                    'rmse': float(result['rmse']),
                    'r2': float(result['r2'])
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(model_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 4. 生成模型說明文件 (README.md)
            self.save_model_readme(model_dir, config, result)
            
            # 5. 保存最佳增強後的數據
            if result['augmentation'] != 'None':
                print(f"  正在為第{i}名模型生成增強數據文件...")
                
                X_base = X_sample
                y_base = y_sample
                
                aug_data_X = [X_base]
                aug_data_y = [y_base]
                
                aug_func = self.augmentations[result['augmentation']]
                current_aug_factor = result.get('aug_factor', aug_factor)
                
                for _ in range(current_aug_factor):
                    X_aug, y_aug = aug_func(X_base, y_base)
                    aug_data_X.append(X_aug)
                    aug_data_y.append(y_aug)
                
                X_final = np.vstack(aug_data_X)
                y_final = np.vstack(aug_data_y)
                
                in_cols = [f"Input_{j}" for j in range(X_sample.shape[1])]
                out_cols = [f"Output_{j}" for j in range(y_sample.shape[1])]
                
                df_aug = pd.DataFrame(X_final, columns=in_cols)
                for idx, col in enumerate(out_cols):
                    df_aug[col] = y_final[:, idx]
                
                save_path = os.path.join(model_dir, f'augmented_data_{current_aug_factor}x.xlsx')
                df_aug.to_excel(save_path, index=False)
                print(f"  已保存增強數據至: {save_path}")

            print(f"  第{i}名: {result['model']} + {result['scaler']} + {result['augmentation']}")
            print(f"         MSE={result['mse']:.6f}, MAPE={result.get('mape',0):.2f}%, Epochs={result.get('actual_epochs',0)}")
    
    def save_model_readme(self, model_dir, config, result):
        """生成模型的說明文件"""
        readme_path = os.path.join(model_dir, 'README.md')
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# Top {config['rank']} Model - {config['model_class']}\n\n")
            f.write(f"## 模型概覽\n")
            f.write(f"- **架構**: {config['model_class']}\n")
            f.write(f"- **數據縮放**: {config['scaler_type']}\n")
            aug_desc = f"{config['augmentation']} (由 {config['aug_factor']} 倍數據訓練)"
            if config.get('aug_include_ori', False):
                aug_desc += " [包含原始數據]"
            f.write(f"- **數據增強**: {aug_desc}\n")
            f.write(f"- **實際訓練輪數**: {config.get('actual_epochs', 'N/A')}\n")
            f.write(f"- **輸入維度**: {config['input_dim']}\n")
            f.write(f"- **輸出維度**: {config['output_dim']}\n\n")
            
            f.write(f"## 性能指標\n")
            f.write(f"| 指標 | 數值 |\n")
            f.write(f"|---|---|\n")
            f.write(f"| MSE | {config['metrics']['mse']:.6f} |\n")
            f.write(f"| MAPE | {config['metrics']['mape']:.2f}% |\n")
            f.write(f"| MAE | {config['metrics']['mae']:.6f} |\n")
            f.write(f"| RMSE | {config['metrics']['rmse']:.6f} |\n")
            f.write(f"| R² | {config['metrics']['r2']:.4f} |\n\n")
            
            f.write(f"## 使用說明\n")
            f.write("```python\n")
            f.write("import torch\n")
            f.write("import joblib\n")
            f.write("from spectra_ae_system import *\n\n")
            f.write("# 1. 加載模型\n")
            f.write(f"input_dim = {config['input_dim']}\n")
            f.write(f"output_dim = {config['output_dim']}\n")
            f.write(f"model = {config['model_class']}(input_dim, output_dim)\n")
            f.write("model.load_state_dict(torch.load('model.pth'))\n")
            f.write("model.eval()\n\n")
            
            if config['scaler_type'] != 'NoScaler':
                f.write("# 2. 加載 Scaler\n")
                f.write("scaler_X = joblib.load('scaler_X.pkl')\n")
                f.write("scaler_y = joblib.load('scaler_y.pkl')\n\n")
                f.write("# 3. 推論\n")
                f.write("input_data = ... # 您的輸入數據\n")
                f.write("input_scaled = scaler_X.transform(input_data)\n")
                f.write("with torch.no_grad():\n")
                f.write("    output_scaled = model(torch.FloatTensor(input_scaled))\n")
                f.write("    output_pred = scaler_y.inverse_transform(output_scaled.numpy())\n")
            else:
                f.write("# 3. 推論\n")
                f.write("input_data = ... # 您的輸入數據\n")
                f.write("with torch.no_grad():\n")
                f.write("    output_pred = model(torch.FloatTensor(input_data)).numpy()\n")
            f.write("```\n")

    def generate_report(self):
        """生成 Markdown benchmark 報告"""
        report_path = os.path.join(self.save_dir, 'benchmark_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 光譜轉換 AutoEncoder 訓練報告\n\n")
            f.write(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. 實驗概況\n")
            f.write(f"- **總共測試組合數**: {len(self.results)}\n")
            if self.results:
                f.write(f"- **最佳 MAPE**: {self.results[0].get('mape',0):.2f}%\n")
                f.write(f"- **對應以及R²**: {self.results[0]['r2']:.4f}\n")
                f.write(f"- **對應以及MSE**: {self.results[0]['mse']:.6f}\n\n")
            
            # 前10名結果表
            f.write("## 2. 前10名最佳模型 (依 MAPE 排序)\n\n")
            f.write("| Rank | Full Name (Model + Scaler + Aug) | MAPE | Epochs | MSE | R² | MAE |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for i, result in enumerate(self.results[:10], 1):
                full_name = f"{result['model']} + {result['scaler']} + {result['augmentation']}"
                epochs = result.get('actual_epochs', 'N/A')
                f.write(f"| {i} | {full_name} | {result.get('mape',0):.2f}% | {epochs} | {result['mse']:.6f} | {result['r2']:.4f} | {result['mae']:.6f} |\n")
            f.write("\n")
            
            # 各模型架構的平均性能 (使用 MAPE)
            f.write("## 3. 各模型架構平均性能 (MAPE)\n\n")
            f.write("| Model | Avg MAPE | Best MAPE |\n")
            f.write("|---|---|---|\n")
            
            model_stats = {}
            for result in self.results:
                m = result['model']
                if m not in model_stats: model_stats[m] = []
                model_stats[m].append(result.get('mape', 0))
            
            for model, mapes in sorted(model_stats.items(), key=lambda x: np.mean(x[1])):
                f.write(f"| {model} | {np.mean(mapes):.2f}% | {np.min(mapes):.2f}% |\n")
            f.write("\n")
            
            # 各縮放器平均性能
            f.write("## 4. 各數據縮放器平均性能 (MAPE)\n\n")
            f.write("| Scaler | Avg MAPE | Best MAPE |\n")
            f.write("|---|---|---|\n")
            scaler_stats = {}
            for result in self.results:
                s = result['scaler']
                if s not in scaler_stats: scaler_stats[s] = []
                scaler_stats[s].append(result.get('mape', 0))
            for scaler, mapes in sorted(scaler_stats.items(), key=lambda x: np.mean(x[1])):
                f.write(f"| {scaler} | {np.mean(mapes):.2f}% | {np.min(mapes):.2f}% |\n")
            f.write("\n")
            
        print(f"\nMarkdown 報告已保存至: {report_path}")

    def generate_pdf_report(self):
        """生成 PDF 報告 (包含圖表)"""
        # 修正: Unicode minus warn
        plt.rcParams['axes.unicode_minus'] = False 
        
        pdf_path = os.path.join(self.save_dir, 'benchmark_report.pdf')
        print("正在生成 PDF 報告...")
        
        with PdfPages(pdf_path) as pdf:
            # 頁面 1: 摘要
            plt.figure(figsize=(10, 12))
            plt.axis('off')
            plt.title("Spectra AE Benchmark Summary", fontsize=20, pad=20)
            
            summary_text = (
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"Total Models Tested: {len(self.results)}\n"
            )
            if self.results:
                best = self.results[0]
                summary_text += (
                    f"\nBest Model (by min MAPE):\n"
                    f"  Full Name:    {best['model']} + {best['scaler']} + {best['augmentation']}\n"
                    f"  MAPE:         {best.get('mape', 0):.2f}%\n"
                    f"  MSE:          {best['mse']:.6f}\n"
                    f"  R2 Score:     {best['r2']:.4f}\n"
                    f"  Actual Epochs:{best.get('actual_epochs', 'N/A')}\n"
                )
            
            plt.text(0.1, 0.8, summary_text, fontsize=12, family='monospace', verticalalignment='top')
            pdf.savefig()
            plt.close()
            
            # 頁面 2: Top 10 Models MAPE Comparison (Bar Chart) - 改為從小到大
            if self.results:
                top_10 = self.results[:10]
                names = [f"#{i}\n{r['model']}\n{r['scaler']}\n{r['augmentation']}" for i, r in enumerate(top_10, 1)]
                mapes = [r.get('mape', 0) for r in top_10]
                
                plt.figure(figsize=(12, 8))
                # Bar chart for MAPE should typically show smallest is best. 
                # Doing horizontal bar: invert axis to put Rank #1 at top.
                bars = plt.barh(names[::-1], mapes[::-1], color='lightgreen')
                plt.xlabel('MAPE (%)')
                plt.title('Top 10 Models - MAPE (Lower is Better)')
                plt.tick_params(axis='y', labelsize=8)
                plt.bar_label(bars, fmt='%.2f')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            
            # 頁面 3: 各模型架構 MAPE 比較 (Box Plot)
            model_mapes = {}
            for r in self.results:
                if r['model'] not in model_mapes: model_mapes[r['model']] = []
                model_mapes[r['model']].append(r.get('mape', 0))
            
            plt.figure(figsize=(10, 6))
            if model_mapes:
                plt.boxplot(list(model_mapes.values()), labels=list(model_mapes.keys()))
                plt.ylabel('MAPE (%)')
                plt.title('MAPE Distribution by Architecture')
                plt.yscale('log') # Log scale 
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # 頁面 4: 各縮放器 MAPE 比較 (Box Plot)
            scaler_mapes = {}
            for r in self.results:
                if r['scaler'] not in scaler_mapes: scaler_mapes[r['scaler']] = []
                scaler_mapes[r['scaler']].append(r.get('mape', 0))
            
            plt.figure(figsize=(10, 6))
            if scaler_mapes:
                plt.boxplot(list(scaler_mapes.values()), labels=list(scaler_mapes.keys()))
                plt.ylabel('MAPE (%)')
                plt.title('MAPE Distribution by Scaler')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # 頁面 5: 各增強方法 MAPE 比較 (Box Plot)
            aug_mapes = {}
            for r in self.results:
                if r['augmentation'] not in aug_mapes: aug_mapes[r['augmentation']] = []
                aug_mapes[r['augmentation']].append(r.get('mape', 0))
            
            plt.figure(figsize=(10, 6))
            if aug_mapes:
                plt.boxplot(list(aug_mapes.values()), labels=list(aug_mapes.keys()))
                plt.ylabel('MAPE (%)')
                plt.title('MAPE Distribution by Augmentation Method')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        print(f"PDF 報告已保存至: {pdf_path}")


# ==================== 使用範例 ====================
if __name__ == "__main__":
    import sys
    import argparse
    
    # 檢查是否使用 CLI 模式 (如果參數大於 1，則啟用 CLI)
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Spectra AE System CLI')
        
        # 數據來源 (二選一)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--file', type=str, help='Path to Excel/CSV data file')
        group.add_argument('--simulate', action='store_true', help='Use simulated data')
        
        # 訓練參數
        parser.add_argument('--aug_factor', type=int, default=6, help='Augmentation factor (default: 6)')
        parser.add_argument('--include_original', action='store_true', help='Include original data in training set during augmentation')
        parser.add_argument('--epochs', type=int, default=500, help='Max training epochs (default: 500)')
        parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (default: 50)')
        parser.add_argument('--save_dir', type=str, default='spectra_ae_results', help='Directory to save results')
        
        args = parser.parse_args()
        
        print("=" * 70)
        print("Spectra AE System - CLI Mode")
        print("=" * 70)
        
        trainer = SpectraAETrainer(save_dir=args.save_dir)
        
        try:
            if args.file:
                X, y, _, _ = trainer.load_data_from_file(args.file)
            else:
                # 模擬數據
                np.random.seed(1688)
                n_samples = 200
                print(f"生成模擬數據: {n_samples} 筆...")
                X = np.random.rand(n_samples, 18) * 0.8 + 0.1
                y = np.zeros((n_samples, 26))
                y[:, :8] = np.random.rand(n_samples, 8) * 0.6 + 0.2
                y[:, 8:] = X + np.random.randn(n_samples, 18) * 0.05
                
            trainer.train_all_combinations(
                X, y, 
                aug_factor=args.aug_factor,
                aug_include_ori=args.include_original,
                epochs=args.epochs,
                patience=args.patience
            )
            
            print("\n" + "=" * 70)
            print("程式執行完成")
            print(f"結果保存在: {trainer.save_dir}")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    else:
        # Interactive Mode (保留原有邏輯)
        trainer = SpectraAETrainer(save_dir='spectra_ae_results')
        
        print("=" * 70)
        print("Spectra Light Transfer AutoEncoder System")
        print("=" * 70)
        print("1. 生成模擬光譜數據 (Simulated Data)")
        print("2. 讀取外部檔案 (Excel/CSV)")
        print("-" * 60)
        
        try:
            choice = input("請選擇 (1/2): ").strip()
            
            if choice == '2':
                # 檔案選擇
                import tkinter as tk
                from tkinter import filedialog
                
                root = tk.Tk()
                root.withdraw()
                
                file_path = filedialog.askopenfilename(
                    title="選擇數據檔案",
                    filetypes=[("Excel Files", "*.xlsx;*.xls"), ("CSV Files", "*.csv")]
                )
                
                if not file_path:
                    print("未選擇檔案，程序退出")
                    sys.exit()
                    
                X, y, _, _ = trainer.load_data_from_file(file_path)
                
            else:
                # 生成模擬數據 (Fallback)
                np.random.seed(1688)
                n_samples = 200
                print(f"生成模擬數據: {n_samples} 筆...")
                X = np.random.rand(n_samples, 18) * 0.8 + 0.1
                y = np.zeros((n_samples, 26))
                y[:, :8] = np.random.rand(n_samples, 8) * 0.6 + 0.2
                y[:, 8:] = X + np.random.randn(n_samples, 18) * 0.05
            
            
            # 詢問增強倍數
            print("-" * 60)
            aug_input = input("請輸入數據增強倍數 (默認 6): ").strip()
            try:
                aug_factor = int(aug_input) if aug_input else 6
            except ValueError:
                print("輸入無效，使用默認值 6")
                aug_factor = 6
            
            # 詢問是否包含原始數據
            ori_input = input("增強時是否包含原始數據? (y/N): ").strip().lower()
            aug_include_ori = (ori_input == 'y' or ori_input == 'yes')
            
            # 詢問 Epochs
            epochs_input = input("請輸入最大訓練 Epochs (默認 500): ").strip()
            try:
                epochs = int(epochs_input) if epochs_input else 500
            except ValueError:
                print("輸入無效，使用默認值 500")
                epochs = 500
                
            # 詢問 Patience
            patience_input = input("請輸入 Early Stopping Patience (默認 50): ").strip()
            try:
                patience = int(patience_input) if patience_input else 50
            except ValueError:
                print("輸入無效，使用默認值 50")
                patience = 50
                
            print(f"設定: 增強={aug_factor}x, 包含原始={aug_include_ori}, Epochs={epochs}, Patience={patience}")
            
            trainer.train_all_combinations(
                X, y, 
                aug_factor=aug_factor, 
                aug_include_ori=aug_include_ori,
                epochs=epochs, 
                patience=patience
            )
            
        except Exception as e:
            print(f"\n發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            input("按 Enter 鍵退出...")
            
        print("\n" + "=" * 70)
        print("程式執行完成")
        print(f"結果保存在: {trainer.save_dir}")
        print("=" * 70)
