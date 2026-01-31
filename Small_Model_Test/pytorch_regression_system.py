# -*- coding: utf-8 -*-
"""Enhanced Deep Learning Regression System with PyTorch

Improvements:
1. Export reports to report.md
2. Show top 20  models in bar charts
3. Add error bars to augmentation performance
4. Replace TensorFlow with PyTorch + CUDA
5. Export augmented datasets
6. Record early stop epochs + configurable patience
7. Memory-efficient model storage
8. CLI mode

Usage Example
Basic Run with File:

python pytorch_regression_system_v2.2.py --file "data.xlsx" --output_dir "my_results"
With Auto-detection (Data has empty separator column):

python pytorch_regression_system_v2.2.py --file "data_with_blank_col.csv"
Explicit Columns:

python pytorch_regression_system_v2.2.py --file "data.csv" --features "A,B,C" --targets "Y1,Y2"
Simulation Mode:

python pytorch_regression_system_v2.2.py --simulate --aug_count 5

Author: BK Liao 
Date: 2026.01.29 v2.3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr, spearmanr
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Check CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ============================================================================
# PyTorch Model Architectures
# ============================================================================

class DeepFFN(nn.Module):
    """Deep Feedforward Network"""
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 128, 64, 32]):
        super(DeepFFN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class WideNetwork(nn.Module):
    """Wide Network with fewer layers"""
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128]):
        super(WideNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4 if i == 0 else 0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.block(x) + x)

class ResNet(nn.Module):
    """Residual Network"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.res_block = ResidualBlock(hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block(x)
        x = self.res_block(x)
        x = self.res_block(x)
        return self.output_layer(x)

class EnsembleNet(nn.Module):
    """Ensemble-like Network with multiple pathways"""
    def __init__(self, input_dim, output_dim, path1_dim=32, path2_dim=64):
        super(EnsembleNet, self).__init__()
        # Pathway 1: Deep and narrow
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, path1_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(path1_dim, path1_dim // 2),
            nn.ReLU()
        )
        # Pathway 2: Shallow and wide
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, path2_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Combine
        combined_dim = path1_dim // 2 + path2_dim
        self.output = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        combined = torch.cat([p1, p2], dim=1)
        return self.output(combined)

class AutoEncoderNet(nn.Module):
    """Autoencoder-style Network"""
    def __init__(self, input_dim, output_dim, bottleneck_dim=16):
        super(AutoEncoderNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

# ============================================================================
# Early Stopping Class
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=30, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        self.stopped_epoch = 0
        self.best_epoch = 0

    
    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
            self.best_epoch = epoch

# ============================================================================
# PyTorch Training Function
# ============================================================================

def train_pytorch_model(model, X_train, y_train, X_val, y_val, 
                       epochs=300, batch_size=32, lr=0.001, patience=30):
    """Train PyTorch model with early stopping"""
    
    # Prepare data
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.FloatTensor(y_train).to(DEVICE)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).to(DEVICE),
        torch.FloatTensor(y_val).to(DEVICE)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping check
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            break
        
        model.train()
    
    # Restore best model
    model.load_state_dict(early_stopping.best_model)
    
    return model, (early_stopping.stopped_epoch,early_stopping.best_epoch) if early_stopping.early_stop else (epochs,epochs)


# --- Calculate MAPE (mean and std) using Predict2 ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_idx = y_true != 0
    mape = np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx]) * 100
    return mape

# ============================================================================
# Main Regressor Class
# ============================================================================

class EnhancedSmallDatasetRegressor:
    """Enhanced regressor with PyTorch and memory optimization"""
    
    def __init__(self, test_size=0.05, validation_split=0.15, early_stop_patience=30,
                 use_combined_augmentation=False, n_augmentations=1, include_original=False):
        """
        Initialize regressor
        
        Parameters:
        - test_size: Proportion of test data
        - validation_split: Proportion of validation data
        - early_stop_patience: Patience for early stopping
        - use_combined_augmentation: If True, combine all augmented data into one large dataset
                                     If False, create separate augmented variants for comparison
        - n_augmentations: Number of augmentations (meaning depends on strategy)
        """
        self.test_size = test_size
        self.validation_split = validation_split
        self.early_stop_patience = early_stop_patience
        self.use_combined_augmentation = use_combined_augmentation
        self.n_augmentations = n_augmentations
        self.include_original_in_aug = include_original
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_model_byMAPE = None
        self.data_info = {}
        self.feature_names = []
        self.target_names = []
        self.augmented_datasets = {}
        self.report_lines = []
        self.training_history = {}
    
    def log(self, message):
        """Log message to both console and report"""
        print(message)
        self.report_lines.append(message)
    
    def save_report(self, filename="report.md", save_path=None, silent=False):
        """Save report to markdown file"""
        if save_path:
            filepath = os.path.join(save_path, filename) if os.path.isdir(save_path) else save_path
        else:
            root = tk.Tk()
            root.withdraw()
            
            filepath = filedialog.asksaveasfilename(
                title="保存報告",
                defaultextension=".md",
                filetypes=[("Markdown files", "*.md"), ("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=filename,
                initialdir=os.path.expanduser("~\\Downloads")
            )
            
            root.destroy()
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Regression Analysis Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write("\n".join(self.report_lines))
            
            self.log(f" Report saved to: {filepath}")
            if not silent:
                messagebox.showinfo("成功", f"報告已保存至:\n{filepath}")
    
    def load_data_from_upload(self, file_path=None, silent=False):
        """Load data from CSV or Excel"""
        
        if file_path:
            filename = file_path
        else:
            self.log("請選擇您的 CSV 或 Excel 檔案:")
            
            root = tk.Tk()
            root.withdraw()
            
            filetypes = [
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title="選擇資料檔案",
                filetypes=filetypes,
                initialdir=os.path.expanduser("~\\Downloads")
            )
            
            root.destroy()
        
        if not filename:
            self.log("未選擇檔案，操作已取消。")
            return None, None
        
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(filename, encoding='utf-8')
            elif filename.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(filename)
            else:
                raise ValueError("請選擇 CSV 或 Excel 檔案")
            
            self.log(f"\n Data loaded successfully!")
            self.log(f"Shape: {data.shape}")
            self.log(f"Columns: {list(data.columns)}")
            
            return data, os.path.basename(filename)
        
        except Exception as e:
            self.log(f" 載入資料時發生錯誤: {e}")
            if not silent:
                messagebox.showerror("錯誤", f"載入資料時發生錯誤: {e}")
            return None, None
    
    def prepare_regression_data(self, data, target_columns=None, feature_columns=None, silent=False):
        """Prepare regression data"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found")
        
        self.log(f"\nAvailable numeric columns: {numeric_columns}")
        
        if target_columns is None:
            if silent:
                # In silent mode, try auto-detect (empty column), otherwise default to last column
                # Logic copied from Option 3 then Option 1
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                split_index = -1
                for i, col in enumerate(numeric_cols):
                     if data[col].isna().all() or 'Unnamed' in col: 
                         if data[col].isna().all():
                             split_index = i
                             break
                
                if split_index != -1:
                    self.log(f"Auto-detected delimiter column: {numeric_cols[split_index]}")
                    target_columns = numeric_cols[split_index+1:]
                else:
                    self.log("No delimiter found, defaulting to last numeric column as target.")
                    target_columns = [numeric_columns[-1]]
            else:
                print("\nTarget column selection:")
                print("1. Single target (last numeric column)")
                print("2. Multiple targets (specify)")
                print("3. Auto-detect (empty column as the delimiter)")
                
                choice = input("Choose option (1/2/3): ").strip()
                
                if choice == "1":
                    target_columns = [numeric_columns[-1]]
                elif choice == "2":
                    target_input = input("Enter target column names (comma-separated): ").strip()
                    target_columns = [col.strip() for col in target_input.split(",")]
                else:
                    # Auto-detection logic (Empty Column Delimiter)
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    split_index = -1
                
                    # Search for empty column
                    for i, col in enumerate(numeric_cols):
                        if data[col].isna().all():
                            split_index = i
                            print(f"Detected empty column '{col}' as delimiter.")
                            break
                    
                    if split_index != -1:
                        target_columns = numeric_cols[:split_index]
                        feature_columns = numeric_cols[split_index+1:] # Skip the empty column
                        
                        # Remove empty column from data if needed, but select specific columns avoids issue
                    else:
                        # Fallback to default logic handled in prepare_regression_data if None passed
                        self.log("No empty column delimiter found. using default split logic.")
                        self.log("Using last column as target.")
                        target_columns = [numeric_columns[-1]]
        
        target_columns = [col for col in target_columns if col in numeric_columns]
        
        if feature_columns is None:
            feature_columns = [col for col in numeric_columns if col not in target_columns]
        
        self.log(f"\n Selected features: {feature_columns}")
        self.log(f" Selected targets: {target_columns}")
        
        X = data[feature_columns].values.astype(np.float32)
        y = data[target_columns].values.astype(np.float32)
        
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X = X[valid_indices]
        y = y[valid_indices]
        
        self.feature_names = feature_columns
        self.target_names = target_columns
        self.data_info = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_targets': y.shape[1] if len(y.shape) > 1 else 1
        }
        
        self.log(f"\n Regression data prepared:")
        self.log(f"Samples: {len(X)}, Features: {X.shape[1]}, Targets: {y.shape[1] if len(y.shape) > 1 else 1}")
        
        return X, y
    
    def create_augmented_datasets(self, X, y, n_augmentations=3, augment_per_method=True, include_original=False):
        """
        Create augmented datasets with two strategies:
        
        Strategy 1 (augment_per_method=True, default):
        - Create separate augmented versions for each method
        - Used for testing which augmentation works best
        - Each method creates one variant of the original data
        
        Strategy 2 (augment_per_method=False):
        - Apply each augmentation method n_augmentations times
        - Combine all augmented data into ONE large dataset
        - Actually increases data quantity (X.shape[0] * n_augmentations)
        
        Parameters:
        - n_augmentations: How many times to apply each method (Strategy 2)
                          or max number of methods to use (Strategy 1)
        - augment_per_method: If True, use Strategy 1; if False, use Strategy 2
        """
        
        self.log(f"\n Creating augmented datasets with {n_augmentations}x multiplication...")
        self.log(f"Strategy: {'Separate variants' if augment_per_method else 'Combined expansion'}")
        self.log(f"Original data size: {X.shape[0]} samples")

        augmentation_methods = {
            'gaussian_noise_light': lambda x, y_val: (x + np.random.normal(0, 0.02 * np.std(x, axis=0), x.shape), y_val),
            'gaussian_noise_moderate': lambda x, y_val: (x + np.random.normal(0, 0.05 * np.std(x, axis=0), x.shape), y_val),
            'gaussian_noise_heavy': lambda x, y_val: (x + np.random.normal(0, 0.1 * np.std(x, axis=0), x.shape), y_val),
            'feature_scaling_light': lambda x, y_val: (x * np.random.uniform(0.95, 1.05, x.shape[1]), y_val),
            'feature_scaling_moderate': lambda x, y_val: (x * np.random.uniform(0.9, 1.1, x.shape[1]), y_val),
            'mixup': self._mixup_augment,
            'bootstrap': lambda x, y_val: self._bootstrap_augment(x, y_val),
            'feature_dropout': self._feature_dropout_augment,
            'smote_style': self._smote_style_augment,
            'synthetic_interpolation': self._synthetic_interpolation_augment
        }

        
        self.augmented_datasets = {}
        # Always include original dataset to compare with augmented datasets
        self.augmented_datasets['original'] = (X, y)

        if augment_per_method:

            # Always include original dataset
            #self.augmented_datasets['original'] = (X.copy(), y.copy())
            #self.log(f"  ✓ original: Shape {X.shape}")
            
            # For each augmentation method, create n_augmentations times the data
            for method_name, method_func in augmentation_methods.items():
                try:
                    if include_original:
                        # Start with original data into all augmented datasets
                        X_combined = [X.copy()]
                        y_combined = [y.copy()]
                    else:
                        X_combined = []
                        y_combined = []

                    # Generate n_augmentations augmented versions
                    for i in range(n_augmentations):
                        X_aug, y_aug = method_func(X.copy(), y.copy())
                        X_combined.append(X_aug)
                        y_combined.append(y_aug)
                    
                    # Combine all augmented data (original + n_augmentations)
                    X_final = np.vstack(X_combined)
                    y_final = np.vstack(y_combined) if len(y.shape) > 1 else np.concatenate(y_combined)
                    #y_final = np.vstack([y.reshape(-1, 1) for y in y_combined]).flatten()
                    
                    self.augmented_datasets[method_name] = (X_final, y_final)
                    self.log(f"   {method_name}: Shape {X_final.shape} ({n_augmentations}x augmented)")
                    
                except Exception as e:
                    self.log(f"   {method_name}: {str(e)}")
            
            total_datasets = len(self.augmented_datasets)
            if include_original:
                self.log(f"\n Created {total_datasets} datasets ({total_datasets-1} augmented methods, original is included)")
            else:
                self.log(f"\n Created {total_datasets} datasets ({total_datasets-1} augmented methods, original is not included)")
            self.log(f"   Each augmented method contains {n_augmentations}x the original data size")
        

        else:
            # Strategy 2: Combined augmentation for data expansion
            self.log(f"Applying {n_augmentations} augmentations per method...")

            # Start with original data
            X_combined = [X.copy()]
            y_combined = [y.copy()]
            
            # Apply each augmentation method multiple times
            aug_methods_only = {k: v for k, v in augmentation_methods.items() if k != 'original'}
            
            for method_name, method_func in aug_methods_only.items():
                for i in range(n_augmentations):
                    try:
                        X_aug, y_aug = method_func(X.copy(), y.copy())
                        X_combined.append(X_aug)
                        y_combined.append(y_aug)
                        self.log(f"   {method_name} iteration {i+1}: +{X_aug.shape[0]} samples")
                    except Exception as e:
                        self.log(f"   {method_name} iteration {i+1}: {str(e)}")
                        break
            
            # Combine all augmented data
            X_final = np.vstack(X_combined)
            y_final = np.vstack(y_combined) if len(y.shape) > 1 else np.concatenate(y_combined)
            
            self.augmented_datasets['combined_augmented'] = (X_final, y_final)
            self.log(f"\n   Total combined dataset: {X_final.shape[0]} samples (expanded {X_final.shape[0]/X.shape[0]:.1f}x)")


        return self.augmented_datasets
    

    def _mixup_augment(self, X, y):
        """Mixup augmentation - 隨機混合其他資料，並且隨機分配權重(beta distribution)"""
        indices = np.random.permutation(X.shape[0])
        lam = np.random.beta(0.2, 0.2)
        return lam * X + (1 - lam) * X[indices], lam * y + (1 - lam) * y[indices]

    def _synthetic_interpolation_augment(self, X, y):
        """Create synthetic samples by interpolating between existing samples"""
        n_samples = X.shape[0]
        # Randomly select pairs of samples
        idx1 = np.random.randint(0, n_samples, n_samples)
        idx2 = np.random.randint(0, n_samples, n_samples)
        
        # Interpolation weight
        alpha = np.random.uniform(0.3, 0.7, (n_samples, 1))
        
        # Create synthetic samples
        X_synthetic = alpha * X[idx1] + (1 - alpha) * X[idx2]
        
        if len(y.shape) == 1:
            y_synthetic = alpha.flatten() * y[idx1] + (1 - alpha.flatten()) * y[idx2]
        else:
            y_synthetic = alpha * y[idx1] + (1 - alpha) * y[idx2]
        
        return X_synthetic, y_synthetic
    

    def _bootstrap_augment(self, X, y):
        """Bootstrap sampling - 隨機挑選資料(會重複選到同一個資料)"""
        indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        return X[indices], y[indices]
    
    def _feature_dropout_augment(self, X, y):
        """Feature dropout augmentation - randomly set some features to zero"""
        X_aug = X.copy()
        dropout_rate = np.random.uniform(0.1, 0.3)  # Random dropout rate between 10-30%
        n_features_to_drop = int(X.shape[1] * dropout_rate)
        features_to_drop = np.random.choice(X.shape[1], n_features_to_drop, replace=False)
        X_aug[:, features_to_drop] = 0
        return X_aug, y
    
    def _smote_style_augment(self, X, y):
        """SMOTE-style augmentation - create synthetic samples between existing ones"""
        n_samples = X.shape[0]
        n_new_samples = n_samples  # Create same number of new samples
        
        # Randomly select pairs of samples
        indices1 = np.random.choice(n_samples, n_new_samples, replace=True)
        indices2 = np.random.choice(n_samples, n_new_samples, replace=True)
        
        # Create synthetic samples as weighted average
        alpha = np.random.uniform(0.2, 0.8, (n_new_samples, 1))
        X_synthetic = alpha * X[indices1] + (1 - alpha) * X[indices2]
        y_synthetic = alpha * y[indices1] + (1 - alpha) * y[indices2]
        
        # Return only the synthetic samples (original will be added in the main function)
        return X_synthetic, y_synthetic
    
    def design_architectures(self, input_dim, output_dim, data_size):
        """Design PyTorch architectures"""
        architectures = {}
        
        # Deep FFN
        hidden_dims = [max(input_dim*16,min(256, data_size * 2)), max(input_dim*4,min(128, data_size * 2)), max(input_dim*2,min(64, data_size)), min(32, data_size // 2)]
        if input_dim > 20:
            hidden_dims = [input_dim*2] + hidden_dims[::-1] + hidden_dims + [output_dim*2]
        architectures['Deep_FFN'] = lambda: DeepFFN(input_dim, output_dim, hidden_dims)
        
        # Wide Network
        hidden_dims = [min(256, data_size * 8), min(128, data_size * 4)]
        if input_dim > 20:
            hidden_dims = [input_dim*4] + hidden_dims[::-1] + hidden_dims + [output_dim*2]
        architectures['Wide_Network'] = lambda: WideNetwork(input_dim, output_dim, hidden_dims)
        
        # ResNet
        if data_size > 30:
            hidden_dim = min(64, data_size)
            architectures['ResNet'] = lambda: ResNet(input_dim, output_dim, hidden_dim)
        
        # Ensemble
        architectures['Ensemble_Net'] = lambda: EnsembleNet(
            input_dim, output_dim,
            min(32, data_size // 2),
            min(64, data_size)
        )
        
        # AutoEncoder
        if input_dim > 5:
            architectures['AutoEncoder_Net'] = lambda: AutoEncoderNet(
                input_dim, output_dim,
                min(16, data_size // 4)
            )
        
        self.log(f"\n Designed {len(architectures)} architectures")
        return architectures
    
    def get_scalers(self):
        """Get scalers"""
        return {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'NoScaler': None
        }
        

    def train_and_evaluate_models(self, X, y):
        """Train and evaluate all models"""
        self.log("\n Starting model training...")
        
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=1688
        )

        #save X_ori, y_ori from whole ori dataset for Augm test
        X_ori, y_ori = X, y

        augmented_datasets = self.create_augmented_datasets(
            X, y, 
            n_augmentations=self.n_augmentations,
            augment_per_method=not self.use_combined_augmentation
        )
        
        input_dim = X.shape[1]
        output_dim = y.shape[1] if len(y.shape) > 1 else 1
        architectures = self.design_architectures(input_dim, output_dim, len(X))
        scalers = self.get_scalers()
        
        results = {}
        
        for aug_name, (X_aug, y_aug) in augmented_datasets.items():
            self.log(f"\n--- Augmentation: {aug_name} ---")
            
             # For combined augmentation, use all data for training
            if self.use_combined_augmentation and aug_name == 'combined_augmented':
                self.log(f"Using expanded dataset: {X_aug.shape[0]} samples")
                X_aug_temp, _, y_aug_temp, _ = train_test_split(
                    X_aug, y_aug, test_size=self.test_size, random_state=1688
                )
            elif aug_name == 'original':
                X_aug_temp, y_aug_temp = X_temp, y_temp
            else:
                # Check: when N_aug > 1 , this split might not be best ? 
                X_aug_temp, _, y_aug_temp, _ = train_test_split(
                    X_aug, y_aug, test_size=self.test_size, random_state=1688
                )

            for scaler_name, scaler in scalers.items():
                self.log(f"\n  - Scaler: {scaler_name} -  ")
                try:
                    if scaler_name != 'NoScaler':
                        X_train_scaled = scaler.fit_transform(X_aug_temp).astype(np.float32)
                        X_test_scaled = scaler.transform(X_test).astype(np.float32)
                        X_ori_scaled = scaler.fit_transform(X_ori).astype(np.float32)
                    else:
                        X_train_scaled = X_aug_temp.astype(np.float32)
                        X_test_scaled = X_test.astype(np.float32)
                        X_ori_scaled = X_ori.astype(np.float32)
                    
                    # Split train into train/val
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_scaled, y_aug_temp,
                        test_size=self.validation_split, random_state=1688
                    )
                    
                    for arch_name, arch_builder in architectures.items():
                        model_name = f"{arch_name}_{aug_name}_{scaler_name}"
                        
                        try:
                            model = arch_builder().to(DEVICE)
                            
                            # Train
                            model, (stopped_epoch, best_epoch) = train_pytorch_model(
                                model, X_train, y_train, X_val, y_val,
                                epochs=5000,
                                batch_size=min(32, max(4, len(X_train) // 8)),
                                lr=0.001,
                                patience=self.early_stop_patience
                            )

                            # Predict
                            model.eval()
                            with torch.no_grad():
                                y_pred = model(torch.FloatTensor(X_test_scaled).to(DEVICE))
                                y_pred = y_pred.cpu().numpy()

                                # Predict2 - test whole original dataset (good for checking the Augm. model)
                                y_ori_pred = model(torch.FloatTensor(X_ori_scaled).to(DEVICE))
                                y_ori_pred = y_ori_pred.cpu().numpy()
                            

                            # Metrics
                            if len(y_test.shape) == 1 or y_test.shape[1] == 1:
                                y_test_flat = y_test.flatten()
                                y_pred_flat = y_pred.flatten()
                                mae = mean_absolute_error(y_test_flat, y_pred_flat)
                                rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
                                r2 = r2_score(y_test_flat, y_pred_flat)
                            else:
                                mae = np.mean([mean_absolute_error(y_test[:, i], y_pred[:, i])
                                             for i in range(y_test.shape[1])])
                                rmse = np.mean([np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                                              for i in range(y_test.shape[1])])
                                r2 = np.mean([r2_score(y_test[:, i], y_pred[:, i])
                                            for i in range(y_test.shape[1])])
                            # Metrics 2. for MAPE to ori Data, Flatten if needed
                            if len(y_ori.shape) > 1:
                                y_ori_flat = y_ori.reshape(-1, y_ori.shape[-1]) if y_ori.shape[-1] > 1 else y_ori.flatten()
                                y_ori_pred_flat = y_ori_pred.reshape(-1, y_ori_pred.shape[-1]) if y_ori_pred.shape[-1] > 1 else y_ori_pred.flatten()
                            else:
                                y_ori_flat = y_ori.flatten()
                                y_ori_pred_flat = y_ori_pred.flatten()

                            mape_values = mean_absolute_percentage_error(y_ori_flat, y_ori_pred_flat)
                            mape_mean = np.mean(mape_values)
                            mape_std = np.std(mape_values)

                            results[model_name] = {
                                'model_state': model.state_dict(),  # Store state dict only
                                'model_class': arch_name,
                                'scaler': scaler,
                                'architecture': arch_name,
                                'augmentation': aug_name,
                                'scaler_name': scaler_name,
                                'mae': mae,
                                'rmse': rmse,
                                'r2': r2,
                                'mape_mean': mape_mean,
                                'mape_std': mape_std,
                                'predictions': y_pred,
                                'actual': y_test,
                                'stopped_epoch': stopped_epoch,
                                'best_epoch': best_epoch
                            }
                            
                            self.log(f"  {arch_name}: MAE={mae:.4f}, R2={r2:.4f}, MAPE(ori)={mape_mean:.1f}%, Epoch={best_epoch}/{stopped_epoch}")
                            
                            # Clean up
                            del model
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                        except Exception as e:
                            self.log(f"  X {arch_name}: {str(e)[:100]}")
                
                except Exception as e:
                    self.log(f"   X Scaler {scaler_name}: {str(e)[:100]}")
        
        self.results = results
        
        if results:
            best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
            self.best_model_name = best_model_name

            best_model_byMAPE =  min(results.keys(), key=lambda x: results[x]['mape_mean'])
            self.best_model_byMAPE = best_model_byMAPE
            
            self.log(f"\n Best model: {best_model_name}")
            self.log(f"  MAE: {results[best_model_name]['mae']:.4f}")
            self.log(f"  R2: {results[best_model_name]['r2']:.4f}")
            self.log(f"  MAPE(ori): {results[best_model_name]['mape_mean']:.1f} +/- {results[best_model_name]['mape_std']:.1f}%")
            self.log(f"  Stopped at epoch: {results[best_model_name]['stopped_epoch']}")

            self.log(f"\n Best mode by MAPE to whole ori dataset: {best_model_byMAPE}")
            self.log(f"  MAE: {results[best_model_byMAPE]['mae']:.4f}")
            self.log(f"  R2: {results[best_model_byMAPE]['r2']:.4f}")
            self.log(f"  MAPE(ori): {results[best_model_byMAPE]['mape_mean']:.1f} +/- {results[best_model_byMAPE]['mape_std']:.1f}%")
            self.log(f"  Stopped at epoch: {results[best_model_byMAPE]['stopped_epoch']}")
        

        return results

    def save_results_json(self, folder_path):
        """Save results to JSON file"""
        import json
        
        if not self.results:
            self.log("No results to save.")
            return

        json_path = os.path.join(folder_path, 'results.json')
        
        # Convert numbers to native python types for JSON serialization
        # And exclude non-serializable keys
        serializable_results = {}
        exclude_keys = ['model_state', 'scaler', 'predictions', 'actual']
        
        for k, v in self.results.items():
            serializable_results[k] = {
                key: float(val) if isinstance(val, (np.float32, np.float64, np.int32, np.int64)) else val 
                for key, val in v.items()
                if key not in exclude_keys
            }
            
        # Add summary info
        output_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_info': self.data_info,
            'best_model_name': self.best_model_byMAPE,
            'results': serializable_results
        }
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        self.log(f" Results JSON saved to: {json_path}")


    
    def plot_results(self, save_path=None, show_plot=True):
        """Plot results with improvements"""
        if not self.results:
            self.log("No results to plot")
            return
        
        fig = plt.figure(figsize=(24, 24))
        
        # Sort models by MAE
        sorted_models = sorted(self.results.keys(), key=lambda x: self.results[x]['mae'])
        
        # Select top 15 and bottom 5
        # display_models = sorted_models[:15] + sorted_models[-5:]
        # Selct top 20
        display_models = sorted_models[:20]
        mae_scores = [self.results[m]['mae'] for m in display_models]
        r2_scores = [self.results[m]['r2'] for m in display_models]
        mapes = [self.results[m]['mape_mean'] for m in display_models]
        mapes_std = [self.results[m]['mape_std'] for m in display_models]
        
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(display_models)))
        
        # 1. MAE Performance (Top N)
        ax1 = plt.subplot(3, 3, 1)
        y_pos = np.arange(len(display_models))  # 明確設定 y 軸位置
        ax1.barh(y_pos, mae_scores, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)  # 確保每個位置都有刻度
        ax1.set_yticklabels([m[:25] for m in display_models], fontsize=8)  # 調整字體大小
        ax1.set_xlabel('MAE')
        ax1.set_title('Model Performance (MAE)\nTop 20')
        ax1.invert_yaxis()
        
        # 2. R² Performance (Top N)
        ax2 = plt.subplot(3, 3, 2)
        ax2.barh(y_pos, r2_scores, color=colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([m[:25] for m in display_models], fontsize=8)
        ax2.set_xlabel('R2 Score')
        ax2.set_title('Model Performance (R2)\nTop 20')
        ax2.invert_yaxis()
        
        # 3. MAPE Performance (Top N)
        ax3 = plt.subplot(3, 3, 3)
        ax3.barh(y_pos, mapes, xerr=mapes_std, color=colors, alpha=0.8)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([m[:25] for m in display_models], fontsize=8)
        ax3.set_xlabel('MAPE +/- SD (%)')
        ax3.set_title('Model Performance (MAPE to ori Data)\nTop 20')
        ax3.invert_yaxis()        
        
        # 4. Best Model Predictions
        plt.subplot(3, 3, 4)
        best_result = self.results[self.best_model_name]
        y_actual = best_result['actual']
        y_pred = best_result['predictions']
        
        # Flatten for plotting regardless of dimensions (handle multi-output)
        y_actual_plot = y_actual.flatten()
        y_pred_plot = y_pred.flatten()
        
        plt.scatter(y_actual_plot, y_pred_plot, alpha=0.6)
        min_val = min(np.min(y_actual_plot), np.min(y_pred_plot))
        max_val = max(np.max(y_actual_plot), np.max(y_pred_plot))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Best Model Predictions\n{self.best_model_name[:30]}')

        residuals = y_actual_plot - y_pred_plot
        
        # 4. Residuals (deprecated.)
        '''
        plt.subplot(3, 3, 4)
        if len(y_actual.shape) == 1 or y_actual.shape[1] == 1:
            residuals = y_actual_plot - y_pred_plot
            plt.scatter(y_pred_plot, residuals, alpha=0.6, color='green')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
        '''

        # 5. Architecture Performance
        plt.subplot(3, 3, 5)
        arch_performance = {}
        for model_name, result in self.results.items():
            arch = result['architecture']
            if arch not in arch_performance:
                arch_performance[arch] = []
            arch_performance[arch].append(result['mae'])
        
        arch_names = list(arch_performance.keys())
        arch_mae_avg = [np.mean(arch_performance[arch]) for arch in arch_names]
        arch_mae_std = [np.std(arch_performance[arch]) for arch in arch_names]
        
        plt.bar(arch_names, arch_mae_avg, yerr=arch_mae_std, capsize=5, alpha=0.8, color='orange')
        plt.xlabel('Architecture')
        plt.ylabel('Average MAE')
        plt.title('Architecture Performance')
        plt.xticks(rotation=45, ha='right')
        
        # 6. Augmentation Performance with Error Bars
        plt.subplot(3, 3, 6)
        aug_performance = {}
        for model_name, result in self.results.items():
            aug = result['augmentation']
            if aug not in aug_performance:
                aug_performance[aug] = []
            aug_performance[aug].append(result['mae'])
        
        aug_names = list(aug_performance.keys())
        aug_mae_avg = [np.mean(aug_performance[aug]) for aug in aug_names]
        aug_mae_std = [np.std(aug_performance[aug]) for aug in aug_names]
        
        plt.bar(aug_names, aug_mae_avg, yerr=aug_mae_std, capsize=5, alpha=0.8, color='purple')
        plt.xlabel('Augmentation Method')
        plt.ylabel('Average MAE')
        plt.title('Augmentation Performance\n(with std dev)')
        plt.xticks(rotation=45, ha='right')
        
        # 7. Scaler Performance
        plt.subplot(3, 3, 7)
        scaler_performance = {}
        for model_name, result in self.results.items():
            scaler = result['scaler_name']
            if scaler not in scaler_performance:
                scaler_performance[scaler] = []
            scaler_performance[scaler].append(result['mae'])
        
        scaler_names = list(scaler_performance.keys())
        scaler_mae_avg = [np.mean(scaler_performance[scaler]) for scaler in scaler_names]
        
        plt.bar(scaler_names, scaler_mae_avg, alpha=0.8, color='cyan')
        plt.xlabel('Scaler')
        plt.ylabel('Average MAE')
        plt.title('Scaler Performance')
        plt.xticks(rotation=45, ha='right')
        
        # 8. Error Distribution
        plt.subplot(3, 3, 8)
        if True: # Always plot (residuals already flattened above)
            errors = residuals
            plt.hist(errors, bins=min(15, max(5, len(errors)//3)), alpha=0.7, edgecolor='black', color='red')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.axvline(x=0, color='blue', linestyle='--', alpha=0.7)
        
        # 9. Early Stop Epochs Distribution
        plt.subplot(3, 3, 9)
        epochs = [self.results[m]['stopped_epoch'] for m in self.results.keys()]
        plt.hist(epochs, bins=20, alpha=0.7, edgecolor='black', color='teal')
        plt.xlabel('Stopped Epoch')
        plt.ylabel('Frequency')
        plt.title(f'Early Stop Distribution\n(patience={self.early_stop_patience})')
        plt.axvline(x=np.mean(epochs), color='r', linestyle='--', label=f'Mean: {np.mean(epochs):.0f}')
        plt.legend()
        
        plt.tight_layout(pad=2.0)
        
        if save_path:
            plot_path = os.path.join(save_path, 'results_summary.pdf')
            fig.savefig(plot_path)
            self.log(f" Saved results plot to: {plot_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        # Second figure: Augmentation Analysis
        if self.augmented_datasets:
            fig2 = plt.figure(figsize=(28, 14))
            self._plot_augmentation_analysis()
            
            if save_path:
                aug_plot_path = os.path.join(save_path, 'augmentation_analysis.pdf')
                fig2.savefig(aug_plot_path)
                self.log(f" Saved augmentation analysis to: {aug_plot_path}")
                
            if show_plot:
                plt.show()
            else:
                plt.close(fig2)
        
        self.print_results_table()
    
    def _plot_augmentation_analysis(self):
        """Plot augmentation analysis with PCA, t-SNE, and correlation analysis"""
        if not self.augmented_datasets:
            self.log("No augmented datasets available for analysis")
            return
        
        self.log("\n Creating augmentation analysis plots...")
        
        try:
            # Get original data
            X_original = self.augmented_datasets['original'][0]
            y_original = self.augmented_datasets['original'][1]
            
            # Prepare data for analysis
            datasets_for_analysis = {}
            for name, (X, y) in self.augmented_datasets.items():
                if name == 'original':
                    datasets_for_analysis[name] = (X, y)
                else:
                    # For augmented datasets, sample a subset for visualization
                    n_samples = min(500, len(X))  # Limit to 500 samples for visualization
                    indices = np.random.choice(len(X), n_samples, replace=False)
                    datasets_for_analysis[name] = (X[indices], y[indices])
            
            
            # Create second figure (Removed: use caller's figure)
            # fig = plt.figure(figsize=(28, 14))
            
            # 1. PCA Analysis
            ax1 = plt.subplot(2, 4, 1)
            self._plot_pca_analysis(datasets_for_analysis)
            ax1.set_aspect('auto', adjustable='box')
            
            # 2. t-SNE Analysis
            ax2 = plt.subplot(2, 4, 2)
            self._plot_tsne_analysis(datasets_for_analysis)
            ax2.set_aspect('equal', adjustable='box')
            
            # 3. Correlation Heatmap
            ax3 = plt.subplot(2, 4, 3)
            self._plot_correlation_heatmap(datasets_for_analysis)
            ax3.set_aspect('equal', adjustable='box')
            
            # 4. Feature Distribution Comparison
            ax4 = plt.subplot(2, 4, 4)
            self._plot_feature_distribution(datasets_for_analysis)
            ax4.set_aspect('auto', adjustable='box')
            
            # 5. Target Distribution Comparison
            ax5 = plt.subplot(2, 4, 5)
            self._plot_target_distribution(datasets_for_analysis)
            ax5.set_aspect('auto', adjustable='box')
            
            # 6. Augmentation Similarity Matrix
            ax6 = plt.subplot(2, 4, 6)
            self._plot_similarity_matrix(datasets_for_analysis)
            ax6.set_aspect('equal', adjustable='box')
            
            # 7. Data Quality Metrics
            ax7 = plt.subplot(2, 4, 7)
            self._plot_data_quality_metrics(datasets_for_analysis)
            ax7.set_aspect('auto', adjustable='box')
            
            # 8. Augmentation Effectiveness
            ax8 = plt.subplot(2, 4, 8)
            self._plot_augmentation_effectiveness(datasets_for_analysis)
            ax8.set_aspect('equal', adjustable='box')
            
            plt.tight_layout(pad=2.0)
            #plt.show()
            
        except Exception as e:
            self.log(f" Error in augmentation analysis: {e}")
    
    def _plot_pca_analysis(self, datasets):
        """Plot PCA analysis of different datasets"""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Combine all data for PCA fitting
            all_data = []
            labels = []
            dataset_sizes = []
            colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
            
            for i, (name, (X, y)) in enumerate(datasets.items()):
                all_data.append(X)
                labels.extend([name] * len(X))
                dataset_sizes.append(len(X))
            
            X_combined = np.vstack(all_data)
            
            # 標準化數據
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_combined)
            
            # Fit PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Plot each dataset
            start_idx = 0
            for i, (name, (X, y)) in enumerate(datasets.items()):
                end_idx = start_idx + dataset_sizes[i]
                plt.scatter(X_pca[start_idx:end_idx, 0], X_pca[start_idx:end_idx, 1], 
                        alpha=0.6, s=15, label=name, color=colors[i], edgecolors='white', linewidth=0.3)
                start_idx = end_idx
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=9)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=9)
            plt.title('PCA: Augmentation Distribution', fontsize=10, fontweight='bold')
            plt.legend(fontsize=7, framealpha=0.9, loc='best')
            plt.tick_params(axis='both', which='major', labelsize=7)
            plt.grid(True, alpha=0.3, linestyle='--')
            
        except Exception as e:
            plt.text(0.5, 0.5, f'PCA Error:\n{str(e)[:50]}', 
                    ha='center', va='center', fontsize=7, wrap=True)
            plt.title('PCA Analysis (Error)', fontsize=10)
    
    def _plot_tsne_analysis(self, datasets):
        """Plot t-SNE analysis of different datasets"""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Combine all data for t-SNE
            all_data = []
            labels = []
            dataset_sizes = []
            colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
            
            for i, (name, (X, y)) in enumerate(datasets.items()):
                all_data.append(X)
                labels.extend([name] * len(X))
                dataset_sizes.append(len(X))
            
            X_combined = np.vstack(all_data)
            
            # 標準化數據
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_combined)
            
            # 限制數據量以加速計算
            max_samples = 1000
            if len(X_scaled) > max_samples:
                indices = np.random.choice(len(X_scaled), max_samples, replace=False)
                X_scaled = X_scaled[indices]
                # 更新 dataset_sizes
                new_sizes = []
                start = 0
                for size in dataset_sizes:
                    end = start + size
                    new_size = np.sum((indices >= start) & (indices < end))
                    new_sizes.append(new_size)
                    start = end
                dataset_sizes = new_sizes
            
            # Fit t-SNE with appropriate parameters
            n_samples = len(X_scaled)
            perplexity = min(30, max(5, n_samples // 4))
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity) 
                    #n_iter=1000, learning_rate=200)
            X_tsne = tsne.fit_transform(X_scaled)
            
            # Plot each dataset
            start_idx = 0
            for i, (name, (X, y)) in enumerate(datasets.items()):
                end_idx = start_idx + dataset_sizes[i]
                if end_idx > start_idx:  # 確保有數據點
                    plt.scatter(X_tsne[start_idx:end_idx, 0], X_tsne[start_idx:end_idx, 1], 
                            alpha=0.6, s=15, label=name, color=colors[i], 
                            edgecolors='white', linewidth=0.3)
                start_idx = end_idx
            
            plt.xlabel('t-SNE 1', fontsize=9)
            plt.ylabel('t-SNE 2', fontsize=9)
            plt.title('t-SNE: Augmentation Distribution', fontsize=10, fontweight='bold')
            plt.legend(fontsize=7, framealpha=0.9, loc='best')
            plt.tick_params(axis='both', which='major', labelsize=7)
            plt.grid(True, alpha=0.3, linestyle='--')
            
        except Exception as e:
            plt.text(0.5, 0.5, f't-SNE Error:\n{str(e)[:50]}', 
                    ha='center', va='center', fontsize=7)
            plt.title('t-SNE Analysis (Error)', fontsize=10)
    
    def _plot_correlation_heatmap(self, datasets):
        """Plot correlation heatmap between datasets"""
        try:
            X_original = datasets['original'][0]
            correlations = {}
            
            for name, (X, y) in datasets.items():
                if name != 'original':
                    # 確保長度匹配
                    min_len = min(len(X_original), len(X))
                    min_features = min(X_original.shape[1], X.shape[1])
                    
                    # 計算每個特徵的相關係數
                    feature_corrs = []
                    for i in range(min_features):
                        # 使用相同長度的數據計算相關係數
                        try:
                            corr, _ = pearsonr(X_original[:min_len, i], X[:min_len, i])
                            if not np.isnan(corr):  # 過濾 NaN 值
                                feature_corrs.append(abs(corr))  # 使用絕對值
                        except:
                            continue
                    
                    if feature_corrs:
                        correlations[name] = np.mean(feature_corrs)
                    else:
                        correlations[name] = 0.0
            
            if not correlations:
                plt.text(0.5, 0.5, 'No correlations calculated', ha='center', va='center', fontsize=8)
                return
            
            # 創建熱力圖
            names = list(correlations.keys())
            corr_values = list(correlations.values())
            
            vmin_corr = max(0, np.min(corr_values) - 0.1)
            vmax_corr = min(1, np.max(corr_values) + 0.1)
            im = plt.imshow(np.array(corr_values).reshape(1, -1), cmap='RdYlBu_r', 
                        aspect='auto', vmin=vmin_corr, vmax=vmax_corr)
            plt.colorbar(im, shrink=0.8)
            plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=6)
            plt.yticks([0], ['Avg Correlation'], fontsize=6)
            plt.title('Feature Correlation with Original', fontsize=9)
            
            for i, val in enumerate(corr_values):
                plt.text(i, 0, f'{val:.3f}', ha='center', va='center', fontweight='bold', fontsize=6)
            plt.tick_params(axis='both', which='major', labelsize=6)
                
        except Exception as e:
            plt.text(0.5, 0.5, f'Correlation Error: {str(e)[:50]}', ha='center', va='center', fontsize=6)
            plt.title('Correlation Analysis (Error)', fontsize=9)
    
    def _plot_feature_distribution(self, datasets):
        """Plot feature distribution comparison"""
        try:
            n_features_to_plot = min(1, datasets['original'][0].shape[1])
            
            # 為每個特徵創建一個小的子圖區域
            for i in range(n_features_to_plot):
                # 在當前subplot中創建更小的區域
                if n_features_to_plot == 1:
                    pass  # 使用整個subplot
                elif n_features_to_plot == 2:
                    plt.subplot(2, 1, i+1)
                else:
                    plt.subplot(n_features_to_plot, 1, i+1)
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
                
                # 繪製原始數據
                orig_data = datasets['original'][0][:, i]
                plt.hist(orig_data, alpha=0.5, bins=20, 
                        label='Original', density=True, 
                        color=colors[0], edgecolor='black', linewidth=0.3)
                
                # 繪製增強數據
                dataset_idx = 1
                for name, (X, y) in datasets.items():
                    if name != 'original':
                        min_len = min(len(X), len(orig_data))
                        aug_data = X[:min_len, i]
                        plt.hist(aug_data, alpha=0.4, bins=20, 
                                label=name, density=True,
                                color=colors[dataset_idx], edgecolor='black', linewidth=0.3)
                        dataset_idx += 1
                
                plt.ylabel('Density', fontsize=6)
                plt.title(f'Feature {i+1} Distribution', fontsize=7, fontweight='bold')
                if i == 0:
                    plt.legend(fontsize=5, loc='best', framealpha=0.9)
                if i == n_features_to_plot - 1:
                    plt.xlabel('Value', fontsize=6)
                plt.tick_params(axis='both', which='major', labelsize=5)
                plt.grid(True, alpha=0.3)
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Distribution Error:\n{str(e)[:50]}', 
                    ha='center', va='center', fontsize=7)
            plt.title('Feature Distribution (Error)', fontsize=9)
    
    def _plot_target_distribution(self, datasets):
        """Plot target distribution comparison"""
        try:
            colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
            
            # 檢查是分類還是回歸問題
            y_original = datasets['original'][1]
            is_classification = len(np.unique(y_original)) < 20  # 假設少於20個唯一值為分類
            
            if is_classification:
                # 分類問題:使用計數圖
                for idx, (name, (X, y)) in enumerate(datasets.items()):
                    unique, counts = np.unique(y, return_counts=True)
                    plt.bar(unique + idx * 0.15, counts, width=0.15, 
                        label=name, alpha=0.7, color=colors[idx])
                plt.xlabel('Class', fontsize=8)
                plt.ylabel('Count', fontsize=8)
                plt.title('Target Class Distribution', fontsize=9, fontweight='bold')
            else:
                # 回歸問題:使用直方圖
                for idx, (name, (X, y)) in enumerate(datasets.items()):
                    plt.hist(y, alpha=0.5, bins=20, label=name, density=True,
                            color=colors[idx], edgecolor='black', linewidth=0.5)
                plt.xlabel('Target Value', fontsize=8)
                plt.ylabel('Density', fontsize=8)
                plt.title('Target Distribution Comparison', fontsize=9, fontweight='bold')
            
            plt.legend(fontsize=7, framealpha=0.9, loc='best')
            plt.tick_params(axis='both', which='major', labelsize=7)
            plt.grid(True, alpha=0.3, linestyle='--', axis='y')
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Target Distribution Error:\n{str(e)[:50]}', 
                    ha='center', va='center', fontsize=7)
            plt.title('Target Distribution (Error)', fontsize=9)
    
    def _plot_similarity_matrix(self, datasets):
        """Plot similarity matrix between datasets"""
        try:
            dataset_names = list(datasets.keys())
            n_datasets = len(dataset_names)
            similarity_matrix = np.zeros((n_datasets, n_datasets))
            
            for i, name1 in enumerate(dataset_names):
                for j, name2 in enumerate(dataset_names):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        X1 = datasets[name1][0]
                        X2 = datasets[name2][0]
                        min_len = min(len(X1), len(X2))
                        min_features = min(X1.shape[1], X2.shape[1])
                        
                        # 使用餘弦相似度或歐氏距離的相似度
                        # 方法1: 餘弦相似度 (推薦)
                        similarities = []
                        for k in range(min_len):
                            v1 = X1[k, :min_features]
                            v2 = X2[k, :min_features]
                            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                            similarities.append(cos_sim)
                        similarity_matrix[i, j] = np.mean(similarities)
                        
                        # 方法2: 基於歐氏距離的相似度
                        # distances = np.linalg.norm(X1[:min_len, :min_features] - X2[:min_len, :min_features], axis=1)
                        # max_distance = np.sqrt(min_features) * (np.max(X1) - np.min(X1))
                        # similarity_matrix[i, j] = 1 - np.mean(distances) / max_distance
            
            # Plot heatmap
            vmin_val = max(-1, np.min(similarity_matrix) - 0.1)
            vmax_val = min(1, np.max(similarity_matrix) + 0.1)
            im = plt.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=vmin_val, vmax=vmax_val)
            plt.colorbar(im, shrink=0.8)
            plt.xticks(range(n_datasets), dataset_names, rotation=45, ha='right', fontsize=6)
            plt.yticks(range(n_datasets), dataset_names, fontsize=6)
            plt.title('Dataset Similarity Matrix', fontsize=9)
            
            for i in range(n_datasets):
                for j in range(n_datasets):
                    plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                            ha='center', va='center', fontweight='bold', fontsize=6)
            plt.tick_params(axis='both', which='major', labelsize=6)
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Similarity Error: {str(e)[:30]}', ha='center', va='center', fontsize=6)
            plt.title('Similarity Matrix (Error)', fontsize=9)
    
    def _plot_data_quality_metrics(self, datasets):
        """Plot data quality metrics for each dataset"""
        try:
            metrics = {}
            for name, (X, y) in datasets.items():
                # 計算更全面的指標
                metrics[name] = {
                    'Mean': np.mean(np.abs(X)),
                    'Std': np.std(X),
                    'Skewness': np.mean([abs(np.mean((X[:, i] - np.mean(X[:, i]))**3) / (np.std(X[:, i])**3 + 1e-8)) 
                                        for i in range(X.shape[1])]),
                    'Sparsity': np.sum(np.abs(X) < 1e-5) / X.size
                }
            
            # 創建子圖顯示不同指標
            metric_names = list(metrics[list(metrics.keys())[0]].keys())
            x = np.arange(len(datasets))
            width = 0.8 / len(metric_names)
            colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
            
            for i, metric in enumerate(metric_names):
                values = [metrics[name][metric] for name in datasets.keys()]
                plt.bar(x + i * width, values, width, label=metric, alpha=0.8, color=colors[i])
            
            plt.xlabel('Dataset', fontsize=8)
            plt.ylabel('Value', fontsize=8)
            plt.title('Data Quality Metrics', fontsize=9, fontweight='bold')
            plt.xticks(x + width * (len(metric_names) - 1) / 2, list(datasets.keys()), 
                    rotation=45, ha='right', fontsize=6)
            plt.legend(fontsize=6, framealpha=0.9, loc='best')
            plt.tick_params(axis='both', which='major', labelsize=7)
            plt.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # 只在必要時使用 log scale
            if max([metrics[name]['Mean'] for name in datasets.keys()]) / \
            min([metrics[name]['Sparsity'] for name in datasets.keys()]) > 100:
                plt.yscale('log')
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Quality Metrics Error:\n{str(e)[:50]}', 
                    ha='center', va='center', fontsize=7)
            plt.title('Data Quality (Error)', fontsize=9)

    def _plot_augmentation_effectiveness(self, datasets):
        """Plot augmentation effectiveness metrics"""
        try:
            effectiveness = {}
            X_original = datasets['original'][0]
            y_original = datasets['original'][1]
            
            for name, (X, y) in datasets.items():
                if name != 'original':
                    min_len = min(len(X_original), len(X))
                    min_features = min(X_original.shape[1], X.shape[1])
                    
                    # 1. Diversity: 與原始數據的差異度
                    correlations = []
                    for i in range(min_features):
                        try:
                            corr, _ = pearsonr(X_original[:min_len, i], X[:min_len, i])
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                        except:
                            pass
                    diversity = 1 - np.mean(correlations) if correlations else 0.5
                    
                    # 2. Coverage: 數據覆蓋範圍
                    coverage = min(1.0, np.std(X) / (np.std(X_original) + 1e-8))
                    
                    # 3. Balance: 目標分佈的平衡性
                    if len(np.unique(y_original)) < 20:  # 分類問題
                        _, counts_orig = np.unique(y_original, return_counts=True)
                        _, counts_aug = np.unique(y[:min_len], return_counts=True)
                        balance = 1 - np.mean(np.abs(counts_orig / np.sum(counts_orig) - 
                                                    counts_aug / np.sum(counts_aug)))
                    else:  # 回歸問題
                        balance = 1 - abs(np.mean(y) - np.mean(y_original)) / (np.std(y_original) + 1e-8)
                        balance = np.clip(balance, 0, 1)
                    
                    effectiveness[name] = {
                        'Diversity': np.clip(diversity, 0, 1),
                        'Coverage': np.clip(coverage, 0, 1),
                        'Balance': np.clip(balance, 0, 1)
                    }
            
            if not effectiveness:
                plt.text(0.5, 0.5, 'No augmented datasets', ha='center', va='center', fontsize=8)
                return
            
            # 使用普通座標繪製雷達圖風格的圖表
            metric_names = ['Diversity', 'Coverage', 'Balance']
            num_vars = len(metric_names)
            
            # 計算角度
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
            
            ax = plt.gca()
            colors = plt.cm.Set2(np.linspace(0, 1, len(effectiveness)))
            
            # 繪製網格圓
            for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
                circle_x = r * np.cos(np.linspace(0, 2*np.pi, 100))
                circle_y = r * np.sin(np.linspace(0, 2*np.pi, 100))
                ax.plot(circle_x, circle_y, 'k-', alpha=0.1, linewidth=0.5)
                ax.text(0, r, f'{r:.1f}', ha='center', va='bottom', fontsize=5, alpha=0.5)
            
            # 繪製軸線
            for angle in angles:
                ax.plot([0, np.cos(angle)], [0, np.sin(angle)], 'k-', alpha=0.2, linewidth=0.5)
            
            # 繪製數據
            for idx, (name, metrics) in enumerate(effectiveness.items()):
                values = [metrics[metric] for metric in metric_names]
                
                # 計算座標
                xs = [values[i] * np.cos(angles[i]) for i in range(num_vars)]
                ys = [values[i] * np.sin(angles[i]) for i in range(num_vars)]
                
                # 閉合多邊形
                xs.append(xs[0])
                ys.append(ys[0])
                
                ax.plot(xs, ys, 'o-', linewidth=2, label=name, 
                    alpha=0.7, markersize=5, color=colors[idx])
                ax.fill(xs, ys, alpha=0.15, color=colors[idx])
            
            # 添加標籤
            for i, (angle, name) in enumerate(zip(angles, metric_names)):
                x = 1.15 * np.cos(angle)
                y = 1.15 * np.sin(angle)
                ha = 'center'
                if x > 0.1:
                    ha = 'left'
                elif x < -0.1:
                    ha = 'right'
                ax.text(x, y, name, ha=ha, va='center', fontsize=7, fontweight='bold')
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Augmentation Effectiveness', fontsize=9, fontweight='bold', pad=5)
            ax.legend(loc='upper right', fontsize=6, framealpha=0.9)
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Effectiveness Error:\n{str(e)[:50]}', 
                    ha='center', va='center', fontsize=7)
            plt.title('Augmentation Effectiveness (Error)', fontsize=9)

    #########################################################

    def print_results_table(self):
        """Print detailed results table"""
        self.log("\n" + "="*160)
        self.log("DETAILED RESULTS TABLE")
        self.log("="*160)
        
        df_results = []
        for model_name, result in self.results.items():
            df_results.append({
                'Model': model_name[:42],
                'Architecture': result['architecture'],
                'Augmentation': result['augmentation'],
                'Scaler': result['scaler_name'],
                'MAE': f"{result['mae']:.4f}",
                'RMSE': f"{result['rmse']:.4f}",
                'R2': f"{result['r2']:.4f}",
                'MAPE(ori)': f"{result['mape_mean']:.1f}+/-{result['mape_std']:.1f}%",
                'Epoch': result['best_epoch']
            })
        
        df = pd.DataFrame(df_results)
        df = df.sort_values('MAE')
        
        # Convert to string for logging
        table_str = df.head(40).to_string(index=False)
        for line in table_str.split('\n'):
            self.log(line)
        
        if len(df) > 40:
            self.log(f"\n... and {len(df)-40} more models")
        
        self.log(f"\n Best model: {self.best_model_name}")
        self.log(f"  MAE: {self.results[self.best_model_name]['mae']:.4f}")
        self.log(f"  R2: {self.results[self.best_model_name]['r2']:.4f}")
        self.log(f"  MAPE(ori): {self.results[self.best_model_name]['mape_mean']:.1f} +/- {self.results[self.best_model_name]['mape_std']:.1f}%")
        self.log(f"  Stopped at epoch: {self.results[self.best_model_name]['stopped_epoch']}")

        self.log(f"\n Best mode by MAPE to whole ori dataset: {self.best_model_byMAPE}")
        self.log(f"  MAE: {self.results[self.best_model_byMAPE]['mae']:.4f}")
        self.log(f"  R2: {self.results[self.best_model_byMAPE]['r2']:.4f}")
        self.log(f"  MAPE(ori): {self.results[self.best_model_byMAPE]['mape_mean']:.1f} +/- {self.results[self.best_model_byMAPE]['mape_std']:.1f}%")
        self.log(f"  Stopped at epoch: {self.results[self.best_model_byMAPE]['stopped_epoch']}")
    
    def save_model(self, filenames=[], folder_path=None, silent=False):
        """Save multiple models to specified folder"""
        if not filenames:
            self.log("沒有指定要儲存的模型檔案名稱")
            return []
        
        if folder_path:
            folder = folder_path
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
        else:
            # 讓使用者選擇儲存資料夾
            root = tk.Tk()
            root.withdraw()
            
            folder = filedialog.askdirectory(
                title="選擇儲存模型的資料夾",
                initialdir=os.path.expanduser("~\\Downloads")
            )
            
            root.destroy()
        
        if not folder:
            self.log("未選擇儲存資料夾，操作已取消。")
            return []
        
        saved_files = []
        
        # 為每個指定的模型名稱儲存模型
        for model_name in filenames:
            if model_name not in self.results:
                self.log(f" 找不到模型: {model_name}")
                continue
            
            result = self.results[model_name]
            
            # 建立檔案名稱
            safe_filename = model_name.replace('/', '_').replace(' ', '_').replace(':', '_')
            filename = os.path.join(folder, f"{safe_filename}.pkl")
            
            # 重建模型架構
            input_dim = self.data_info['n_features']
            output_dim = self.data_info['n_targets']
            architectures = self.design_architectures(input_dim, output_dim, self.data_info['n_samples'])
            
            model_class = result['model_class']
            model = architectures[model_class]().to(DEVICE)
            model.load_state_dict(result['model_state'])
            
            model_package = {
                'model_state_dict': result['model_state'],
                'model_class': model_class,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'scaler': result['scaler'],
                'model_name': model_name,
                'architecture': result['architecture'],
                'augmentation': result['augmentation'],
                'scaler_name': result['scaler_name'],
                'performance': {
                    'mae': result['mae'],
                    'rmse': result['rmse'],
                    'r2': result['r2'],
                    'mape_ori': result['mape_mean'],
                    'stopped_epoch': result['stopped_epoch']
                },
                'data_info': self.data_info,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(model_package, f)
                
                self.log(f" 模型 {model_name} 已儲存為: {filename}")
                saved_files.append(filename)
                
                # 為每個模型建立使用說明
                self._create_usage_instructions(filename, model_package, silent=silent)
                
            except Exception as e:
                self.log(f" 儲存模型 {model_name} 時發生錯誤: {e}")
                continue
        
        if saved_files:
            if not silent:
                messagebox.showinfo("成功", f"已成功儲存 {len(saved_files)} 個模型至:\n{folder}")
        else:
            if not silent:
                messagebox.showwarning("警告", "沒有成功儲存任何模型")
        
        return saved_files
    
    def save_augmented_datasets(self, folder_path=None, silent=False):
        """Save all augmented datasets"""
        if not self.augmented_datasets:
            self.log("No augmented datasets to save")
            return
        
        if folder_path:
            folder = folder_path
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
        else:
            root = tk.Tk()
            root.withdraw()
            
            folder = filedialog.askdirectory(
                title="選擇儲存增強資料集的資料夾",
                initialdir=os.path.expanduser("~\\Downloads")
            )
            
            root.destroy()
        
        if not folder:
            self.log("未選擇資料夾，操作已取消。")
            return
        
        try:
            for aug_name, (X_aug, y_aug) in self.augmented_datasets.items():
                # Create DataFrame
                df_aug = pd.DataFrame(X_aug, columns=self.feature_names)
                
                if len(y_aug.shape) == 1:
                    df_aug[self.target_names[0]] = y_aug
                else:
                    for i, target_name in enumerate(self.target_names):
                        df_aug[target_name] = y_aug[:, i]
                
                # Save to CSV
                csv_path = os.path.join(folder, f"augmented_{aug_name}.csv")
                df_aug.to_csv(csv_path, index=False)
                self.log(f"  [OK] Saved: {csv_path}")
            
            self.log(f"\n All augmented datasets saved to: {folder}")
            if not silent:
                messagebox.showinfo("成功", f"所有增強資料集已儲存至:\n{folder}")
            
        except Exception as e:
            self.log(f" 儲存增強資料集時發生錯誤: {e}")
            if not silent:
                messagebox.showerror("錯誤", f"儲存增強資料集時發生錯誤: {e}")
    
    def _create_usage_instructions(self, filename, model_package, silent=False):
        """Create usage instructions"""
        instructions = f'''# Usage Instructions for PyTorch Regression Model

## Model Information
- **Model Name**: {model_package['model_name']}
- **Architecture**: {model_package['architecture']}
- **Augmentation**: {model_package['augmentation']}
- **Scaler**: {model_package['scaler_name']}
- **Features**: {model_package['feature_names']}
- **Targets**: {model_package['target_names']}
- **Performance**:
  - MAE: {model_package['performance']['mae']:.4f}
  - RMSE: {model_package['performance']['rmse']:.4f}
  - R2: {model_package['performance']['r2']:.4f}
  - Stopped at Epoch: {model_package['performance']['stopped_epoch']}

## Loading and Using the Model

```python
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Load model package
with open('{filename}', 'rb') as f:
    model_package = pickle.load(f)

# Extract components
scaler = model_package['scaler']
feature_names = model_package['feature_names']
target_names = model_package['target_names']

# Define model architecture (copy from training code)
class {model_package['model_class']}(nn.Module):
    # ... (copy architecture definition from main code)
    pass

# Reconstruct and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = {model_package['model_class']}(
    model_package['input_dim'],
    model_package['output_dim']
).to(device)
model.load_state_dict(model_package['model_state_dict'])
model.eval()

# Prediction function
def predict_new_data(new_data):
    """
    Make predictions on new data
    
    Parameters:
    - new_data: numpy array or DataFrame with features: {model_package['feature_names']}
    
    Returns:
    - predictions: numpy array of predicted values
    """
    # Convert DataFrame to array
    if isinstance(new_data, pd.DataFrame):
        new_data = new_data[feature_names].values
    
    # Ensure correct shape
    if len(new_data.shape) == 1:
        new_data = new_data.reshape(1, -1)
    
    # Scale data
    new_data_scaled = scaler.transform(new_data)
    
    # Predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(new_data_scaled).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    return predictions

# Example usage
# Single prediction
new_sample = np.array([[value1, value2, ...]])  # Replace with actual values
prediction = predict_new_data(new_sample)
print(f"Prediction: {{prediction}}")

# DataFrame prediction
new_df = pd.DataFrame({{
    '{model_package['feature_names'][0]}': [val1, val2, ...],
    # ... add all features
}})
predictions = predict_new_data(new_df)
print(f"Predictions: {{predictions}}")
```

## Requirements
- Python 3.7+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn

## Notes
- Ensure input features match training data order
- Handle missing values before prediction
- Model uses {model_package['device']} for computation
- Early stopping patience: Check training logs for optimal configuration
'''
        
        instructions_filename = os.path.splitext(filename)[0] + "_instructions.md"
        
        try:
            with open(instructions_filename, 'w', encoding='utf-8') as f:
                f.write(instructions)
            
            self.log(f" 使用說明已儲存為: {instructions_filename}")
            
            if not silent:
                root = tk.Tk()
                root.withdraw()
                open_file = messagebox.askyesno(
                    "開啟說明檔案",
                    f"使用說明已儲存至:\n{instructions_filename}\n\n是否要開啟說明檔案？"
                )
                root.destroy()
                
                if open_file:
                    os.startfile(instructions_filename)
        
        except Exception as e:
            self.log(f" 儲存說明檔案時發生錯誤: {e}")

# ============================================================================
# Sample Data Generation
# ============================================================================

def create_sample_regression_data(dataset_type='nonlinear', n_samples=80, 
                                 n_features=5, n_targets=1, noise_level=0.1):
    """Create sample regression datasets"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    if dataset_type == 'linear':
        coeffs = np.random.randn(n_features, n_targets)
        y = X @ coeffs
    elif dataset_type == 'nonlinear':
        y = np.zeros((n_samples, n_targets))
        for i in range(n_targets):
            y[:, i] = (np.sin(X[:, 0]) + np.cos(X[:, 1]) * X[:, 2] +
                      np.tanh(X[:, min(3, n_features-1)]))
    elif dataset_type == 'polynomial':
        y = np.zeros((n_samples, n_targets))
        for i in range(n_targets):
            y[:, i] = X[:, 0]**2 + X[:, 1] * X[:, min(2, n_features-1)]
    else:
        y = np.zeros((n_samples, n_targets))
        for i in range(n_targets):
            y[:, i] = np.sin(X[:, 0] * X[:, min(1, n_features-1)])
    
    noise = np.random.normal(0, noise_level, (n_samples, n_targets))
    y = y + noise
    
    if n_targets == 1:
        y = y.flatten()
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    target_names = [f'target_{i+1}' for i in range(n_targets)] if n_targets > 1 else ['target']
    
    data_dict = {name: X[:, i] for i, name in enumerate(feature_names)}
    
    if n_targets == 1:
        data_dict['target'] = y
    else:
        for i, name in enumerate(target_names):
            data_dict[name] = y[:, i]
    
    return pd.DataFrame(data_dict)

# ============================================================================
# Main Execution
# ============================================================================

def run_comprehensive_regression_analysis():
    """Main function"""
    print(" ENHANCED PYTORCH REGRESSION SYSTEM")
    print("=" * 60)
    print("Features:")
    print("1. PyTorch + CUDA support")
    print("2. Memory-efficient model storage")
    print("3. Configurable early stopping")
    print("4. Export reports to markdown")
    print("5. Save augmented datasets")
    print("=" * 60)
    
    # Get early stopping patience
    try:
        patience = int(input("\nEarly stopping patience (default=30): ").strip() or "30")
    except:
        patience = 30
    
    
    try:
        n_augmentations = int(input("\nNumber of augmentations (default=3): ").strip() or "3")
    except:
        n_augmentations = 3
    
    regressor = EnhancedSmallDatasetRegressor(early_stop_patience=patience, n_augmentations=n_augmentations)
    
    # Load data
    regressor.log("\n STEP 1: Loading Data")
    data, filename = regressor.load_data_from_upload()
    
    if data is None:
        return
    
    # Prepare data
    regressor.log("\n STEP 2: Preparing Data")
    try:
        X, y = regressor.prepare_regression_data(data)
    except Exception as e:
        regressor.log(f" Error: {e}")
        return
    
    # Train models
    regressor.log("\n STEP 3: Training Models")
    try:
        results = regressor.train_and_evaluate_models(X, y)
    except Exception as e:
        regressor.log(f" Error: {e}")
        return
    
    if not results:
        regressor.log(" No models trained successfully")
        return
    
    # Plot results
    regressor.log("\n STEP 4: Visualizations")
    try:
        regressor.plot_results()
    except Exception as e:
        regressor.log(f" Plot error: {e}")
    
    # Save report
    regressor.log("\n STEP 5: Saving Report")
    save_report = input("Save report to markdown? (y/n): ").lower().strip()
    if save_report == 'y':
        regressor.save_report()
    
    # Save model
    regressor.log("\n STEP 6: Saving Best Model")
    save_model = input("Save best model? (y/n): ").lower().strip()
    if save_model == 'y':
        regressor.save_model(filenames=[regressor.best_model_name, regressor.best_model_byMAPE])
    
    # Save augmented datasets
    regressor.log("\n STEP 7: Saving Augmented Datasets")
    save_datasets = input("Save augmented datasets? (y/n): ").lower().strip()
    if save_datasets == 'y':
        regressor.save_augmented_datasets()
    
    regressor.log("\nAnalysis Complete!")
    
    return regressor, results

def quick_regression_example():
    """Quick example with synthetic data"""
    print("QUICK EXAMPLE WITH SYNTHETIC DATA")
    print("=" * 50)
    
    dataset_type = input("Dataset type (linear/nonlinear/polynomial, default=nonlinear): ").strip() or "nonlinear"
    n_samples = int(input("Number of samples (default=60): ").strip() or "60")
    n_features = int(input("Number of features (default=4): ").strip() or "4")
    
    df = create_sample_regression_data(dataset_type, n_samples, n_features, 1)
    print(f"\nCreated {dataset_type} dataset:")
    print(df.head())
    
    try:
        patience = int(input("\nEarly stopping patience (default=30): ").strip() or "30")
    except:
        patience = 30
    
    regressor = EnhancedSmallDatasetRegressor(early_stop_patience=patience)
    
    feature_cols = [col for col in df.columns if 'feature' in col]
    target_cols = [col for col in df.columns if 'target' in col]
    
    X, y = regressor.prepare_regression_data(df, target_cols, feature_cols)
    results = regressor.train_and_evaluate_models(X, y)
    regressor.plot_results()
    
    # Save report
    save_report = input("\nSave report? (y/n): ").lower().strip()
    if save_report == 'y':
        regressor.save_report()
    
    return regressor, results

def run_cli_mode():
    """Run system in CLI mode"""
    parser = argparse.ArgumentParser(description='Enhanced PyTorch Regression System CLI')
    
    # Input/Output
    parser.add_argument('--file', type=str, help='Path to input Excel/CSV file')
    parser.add_argument('--simulate', action='store_true', help='Use synthetic data')
    parser.add_argument('--output_dir', type=str, default='regression_results', help='Directory to save results')
    
    # Data Configuration
    parser.add_argument('--targets', type=str, help='Target column names (comma separated)')
    parser.add_argument('--features', type=str, help='Feature column names (comma separated)')
    
    # Training Configuration
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--aug_count', type=int, default=3, help='Number of augmentations')
    parser.add_argument('--include_original', action='store_true', help='Include original data in augmented sets')
    
    # Saving Options
    parser.add_argument('--no_save_report', action='store_true', help='Do not save report')
    parser.add_argument('--no_save_model', action='store_true', help='Do not save best models')
    parser.add_argument('--save_data', action='store_true', help='Save augmented datasets')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Enhanced PyTorch Regression System - CLI Mode")
    print("=" * 70)
    
    # Initialize regressor
    regressor = EnhancedSmallDatasetRegressor(
        early_stop_patience=args.patience,
        n_augmentations=args.aug_count,
        include_original=args.include_original
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    regressor.log("\n STEP 1: Loading Data")
    if args.file:
        data, filename = regressor.load_data_from_upload(file_path=args.file, silent=True)
    elif args.simulate:
        data = create_sample_regression_data(n_samples=200, n_features=18, n_targets=2)
        filename = "simulated_data"
        regressor.log("Generated simulated data")
    else:
        regressor.log("Error: Must specify --file or --simulate")
        sys.exit(1)
        
    if data is None:
        sys.exit(1)
        
    # 2. Prepare Data (Column Selection)
    regressor.log("\n STEP 2: Preparing Data")
    
    target_cols = None
    feature_cols = None
    
    # Parsing columns
    if args.targets:
        target_cols = [c.strip() for c in args.targets.split(',')]
    if args.features:
        feature_cols = [c.strip() for c in args.features.split(',')]
        
    # Auto-detection logic (Empty Column Delimiter)
    if target_cols is None and feature_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        split_index = -1
        
        # Search for empty column
        for i, col in enumerate(numeric_cols):
             if data[col].isna().all():
                 split_index = i
                 regressor.log(f"Detected empty column '{col}' as delimiter.")
                 break
        
        if split_index != -1:
            feature_cols = numeric_cols[:split_index]
            target_cols = numeric_cols[split_index+1:] # Skip the empty column
            
            # Remove empty column from data if needed, but select specific columns avoids issue
        else:
            # Fallback to default logic handled in prepare_regression_data if None passed
            regressor.log("No empty column delimiter found. using default split logic.")
            pass

    try:
        X, y = regressor.prepare_regression_data(data, target_columns=target_cols, feature_columns=feature_cols, silent=True)
    except Exception as e:
        regressor.log(f" Error preparing data: {e}")
        sys.exit(1)
        
    # 3. Train Models
    regressor.log("\n STEP 3: Training Models")
    try:
        results = regressor.train_and_evaluate_models(X, y)
    except Exception as e:
        regressor.log(f" Error training models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    if not results:
        regressor.log(" No models trained successfully")
        sys.exit(1)
        
    # Plot results
    regressor.log("\n STEP 4: Visualizations")
    try:
        regressor.plot_results(save_path=args.output_dir, show_plot=False)
    except Exception as e:
        regressor.log(f" Plotting failed: {e}")
        
    # 4. Save Results
    # Save Report
    if not args.no_save_report:
        regressor.log("\n STEP 5: Saving Report")
        regressor.save_report(save_path=args.output_dir, silent=True)
        
    # Save Models
    if not args.no_save_model:
        regressor.log("\n STEP 6: Saving Best Models")
        models_to_save = []
        if regressor.best_model_name:
            models_to_save.append(regressor.best_model_name)
        if regressor.best_model_byMAPE and regressor.best_model_byMAPE != regressor.best_model_name:
             models_to_save.append(regressor.best_model_byMAPE)
             
        if models_to_save:
            regressor.save_model(filenames=models_to_save, folder_path=args.output_dir, silent=True)
            
    # Save Augmented Data
    if args.save_data:
        regressor.log("\n STEP 7: Saving Augmented Datasets")
        regressor.save_augmented_datasets(folder_path=args.output_dir, silent=True)
        
    # Save JSON Results (Always in CLI mode for integration)
    regressor.save_results_json(args.output_dir)
        
    regressor.log("\n Analysis Complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli_mode()
    else:
        print("ENHANCED PYTORCH REGRESSION SYSTEM")
        print("Choose an option:")
        print("1. Full analysis with your data")
        print("2. Quick example with synthetic data")
        
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "1":
            regressor, results = run_comprehensive_regression_analysis()
        else:
            regressor, results = quick_regression_example()
        
        print("\n" + "="*60)
        print("SYSTEM READY!")
        print("="*60)