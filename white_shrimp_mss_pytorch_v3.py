# -*- coding: utf-8 -*-
"""
white shrimp - MSS deep learning (PyTorch Version)
v3.0 2026-01-30  

Pytorch version of white_shrimp_mss_deep_learning_v2.py
Feature Enhancements:
1. Model saving uses pickle (weights + scaler).
2. Augmentation (synthetic interpolation) + StandardScaler.
3. EnsembleNet support.
4. Scan Logic: Scans ALL possible combinations of inputs (Power Set), not just pairs.
5. Visualization: "Feature Presence" Heatmap sorted by performance.

匯入訓練資料格式 data label:  
[input data]*n欄, 空白1欄 , [labels]*m欄   

NEW v3.1
Loss Function: Switched from MSELoss to HuberLoss(delta=1.0). This handles outliers better, which is crucial for spectral data regression.
Activation: Switched from ReLU to LeakyReLU(negative_slope=0.01) to prevent "dying ReLU" issues in deeper networks.
Scheduler: Added ReduceLROnPlateau (Factor=0.5, Patience=20) to dynamically adjust learning rate when validation loss stalls.

"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
from sklearn.preprocessing import StandardScaler
import itertools

from tkinter import filedialog
from tkinter import Tk

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#**** 設定區  ****#
#### ANN structure ####
ANN_Epoch = 1200      # 最多訓練多少ANN_Epoch
Epoch_patient = 100   # 當 ? 個Epoch後, val_loss沒有降低則提前停止
Train_repeats = 5     # 整個重複幾次? 用來取平均值
Node_N_ratio = 2      # 節點數量的倍率，放大模型用，建議1,2,4倍
BATCH_SIZE = 128      # 批量大小

# Augmentation Settings
USE_AUGMENTATION = False
AUGMENTATION_FACTOR = 2
INCLUDE_ORIGIN = False



# for MSS
ANN_upLayers = np.array([56, 128, 256, 128, 64, 48, 24, 8]) * Node_N_ratio    #[56, 128, 64, 48, 24, 8]
ANN_upDrops = [0, 0.15, 0.2, 0.2,  0.2, 0.2,0.15, 0]   # Dropout

#### Comparison axis ####
Compa_Axis = 1  # 資料組合方式

#### 資料夾、檔名名稱 ####
root_path = os.path.dirname(os.path.abspath(__file__))
have_test_data = True

root = Tk()
root.withdraw()
root_path = filedialog.askdirectory(initialdir=root_path, title="Select Training data folder (v3.0)")
if not root_path:
    print("No folder selected. Exiting.")
    exit()
    
if not root_path.endswith('/'):
    root_path += '/'

test_data_path = ""
if have_test_data:
    test_data_path = filedialog.askopenfilename(initialdir=root_path, title="Select TEST data file", filetypes=(("xls files", "*.xlsx"), ("all files", "*.*")))
    if not test_data_path:
        have_test_data = False
        print('No test data selected, set to False')

# 是否要進行正規化
to_normalize = True

### Classes & Functions ###

class MSSDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data)
        self.label = torch.FloatTensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class MSSModel(nn.Module):
    def __init__(self, input_dim, output_dim, layer_sizes, dropouts):
        super(MSSModel, self).__init__()
        layers = []
        in_dim = input_dim
        
        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if dropouts[i] > 0 and dropouts[i] < 1:
                layers.append(nn.Dropout(p=dropouts[i]))
            in_dim = size
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EnsembleNet(nn.Module):
    """Ensemble-like Network with multiple pathways"""
    def __init__(self, input_dim, output_dim, path1_dim=32, path2_dim=64):
        super(EnsembleNet, self).__init__()
        # Pathway 1: Deep and narrow
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, path1_dim//2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(path1_dim//2, path1_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(path1_dim, path1_dim // 2),
            nn.LeakyReLU(negative_slope=0.01)
        )
        # Pathway 2: Shallow and wide
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, path2_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2)
        )
        # Combine
        combined_dim = (path1_dim // 2) + path2_dim
        self.output = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        # Concatenate features
        c = torch.cat((p1, p2), dim=1)
        return self.output(c)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.best_model_wts = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        if self.best_model_wts:
            model.load_state_dict(self.best_model_wts)

def synthetic_interpolation(data, label, factor=1, include_origin=True):
    """
    Augments data by interpolating between random pairs.
    New_point = lambda * x1 + (1 - lambda) * x2
    """
    aug_data = []
    aug_label = []
    
    N = len(data)
    
    if include_origin:
        aug_data.append(data)
        aug_label.append(label)
        
    for _ in range(factor):
        # Shuffle indices to pick partners
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        # Generate random weights
        lam = np.random.uniform(0, 1.0, size=(N, 1))
        
        # Interpolate
        new_x = lam * data + (1 - lam) * data[indices]
        new_y = lam * label + (1 - lam) * label[indices]
        
        aug_data.append(new_x)
        aug_label.append(new_y)
        
    return np.vstack(aug_data), np.vstack(aug_label)


def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-7
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def Make_Blank_Fig():
    fig = plt.figure(figsize=(8, 16))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    return fig, ax1, ax2

def Plot_Trained_Fig(history, config_name, combination_name, ax1, ax2):
    # Plotting for individual run (not always saved, might just show last one or custom)
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    
    ax1.plot(history['mape'], color='royalblue', linewidth=1, alpha=0.5)
    ax1.plot(history['val_mape'], color='darkorange', linewidth=1, alpha=0.5)
    ax1.set_title(f'{config_name} MAPE: {combination_name}')
    ax1.set(xlabel='epoch', ylabel='mape')
    ax1.legend(['train', 'validation'], loc='upper left')

    ax2.plot(history['loss'], color='royalblue', linewidth=1, alpha=0.5)
    ax2.plot(history['val_loss'], color='darkorange', linewidth=1, alpha=0.5)
    ax2.set_title(f'{config_name} Loss: {combination_name}')
    ax2.set(xlabel='epoch', ylabel='loss')
    ax2.legend(['train', 'validation'], loc='upper left')
    
    #return fig

def Save_Trained_Fig(fig, config_name, combination_name):
    # sanitize filename
    safe_name = combination_name.replace('+', '_')
    fig.savefig(root_path + f'history_{config_name}_{safe_name}.pdf', format='pdf')
    plt.close(fig)

def check_processable_xls(path):
    xls_paths = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(('.xls', '.xlsx')):
                    xls_path = os.path.join(root, file)
                    xls_paths.append(xls_path)
    elif os.path.isfile(path) and path.endswith(('.xls', '.xlsx')):
        xls_paths.append(path)
    else:
        raise SystemExit("Error: No folder selected or invalid path.")
    return xls_paths


##### MAIN LOGIC START #####

# 1. Load Data
xls_paths = check_processable_xls(root_path)
merged_dfs = {}
for xls in xls_paths:
    if '~$ ' in xls or '_output' in xls or '_preds' in xls or 'merged' in xls: continue
    try:
        data_all = pd.read_excel(xls, sheet_name=None, header=None)
    except:
        continue
    
    for s_name, df_sheet in data_all.items():
        na_cols = df_sheet.columns[df_sheet.isna().any()].tolist()
        if not na_cols: continue
        
        if s_name in merged_dfs: merged_dfs[s_name] = pd.concat([merged_dfs[s_name], df_sheet], axis=0)
        else: merged_dfs[s_name] = df_sheet

# Save merged
with pd.ExcelWriter(root_path + 'merged.xlsx') as writer:
    for sName, dataf in merged_dfs.items():
        dataf.to_excel(writer, sheet_name=sName, header=False, index=False)

# Re-read merged
data_all = pd.read_excel(root_path + 'merged.xlsx', sheet_name=None, header=None)
sheet = pd.ExcelFile(root_path + 'merged.xlsx')
ScanN = len(sheet.sheet_names)
sheet_names_list = sheet.sheet_names

# Test Data Loading
test_data_sheet = None
test_data_all = None
if have_test_data:
    if '.xls' in test_data_path:
        test_data_all = pd.read_excel(test_data_path, sheet_name=None, header=None)
        test_data_sheet = pd.ExcelFile(test_data_path)
    elif '.csv' in test_data_path:
        have_test_data = False

# Results Storage (List of Dicts)
configs = ['MLP', 'Ensemble']
results_store = {cfg: [] for cfg in configs}

# Generate All Combinations (Power Set)
all_combinations_indices = []
for r in range(1, ScanN + 1):
    all_combinations_indices.extend(itertools.combinations(range(ScanN), r))

print(f"Total Combinations to Scan: {len(all_combinations_indices)} per config")


for config_name in configs:
    print(f"\n[[ CONFIGURATION: {config_name} ]]")
    
    for combo_indices in all_combinations_indices:
        combo_names_str = "+".join([sheet_names_list[i] for i in combo_indices])
        # Safe name for files (use indices if too long?) - sticking to names for readability but careful with length
        combo_id_str = "_".join([str(i) for i in combo_indices])
        
        print(f"  > Processing: {combo_names_str} (ID: {combo_id_str})")

        # 1. Merge Data Logic
        # Iterate and merge
        data_parts = []
        label_part = None
        
        test_data_parts = []
        has_test_data_for_combo = True
        
        for idx in combo_indices:
            s_name = sheet_names_list[idx]
            df = data_all.get(s_name)
            empty_col = df.columns[df.isna().any()].tolist()[0]
            
            # Data
            d_part = df.iloc[:, :empty_col].to_numpy()
            data_parts.append(d_part)
            
            # Label - Take from the FIRST sheet in the combo as the "primary" label source?
            # Or assume labels are identical for all sheets (likely for MSS if rows are same samples)
            # Original logic: If Compa_Axis=1 (Horizontal), we want one set of labels.
            # Original pairwise logic: label = x_df...
            if label_part is None:
                label_part = df.iloc[:, empty_col+1:].to_numpy()
            
            # Test Data
            if have_test_data and has_test_data_for_combo:
                 if s_name in test_data_sheet.sheet_names:
                     t_df_full = test_data_all.get(s_name)
                     # Find Empty Col in Test
                     t_empty_cols = t_df_full.columns[t_df_full.isna().any()].tolist()
                     if t_empty_cols:
                         t_empty_col = t_empty_cols[0]
                         t_d_part = t_df_full.iloc[:, :t_empty_col].to_numpy()
                     else:
                         t_d_part = t_df_full.to_numpy()
                     
                     test_data_parts.append(t_d_part)
                 else:
                     has_test_data_for_combo = False
        
        if Compa_Axis == 1:
            # Horizontal Stack
            data = np.hstack(data_parts)
            label = label_part
            if has_test_data_for_combo and test_data_parts:
                test_data = np.hstack(test_data_parts)
            else:
                test_data = None
        else:
            # Vertical Stack logic (Compa_Axis=0) is tricky for combos > 1 if labels differ?
            # Assuming Compa_Axis=1 as per user defaults. 
            # If 0, we can't easily merge N sheets properly without assumption. 
            # Fallback to horizontal for now as it makes most sense for "Multi-Spectral" merging.
            data = np.hstack(data_parts)
            label = label_part
            if has_test_data_for_combo and test_data_parts:
                test_data = np.hstack(test_data_parts)
            else:
                test_data = None


        trainN, inputN = data.shape
        labelN, outputN = label.shape

        # 2. Augmentation & Normalization
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_num = int(len(data) * 0.85)
        
        raw_train_X = data[indices[:split_num]].astype(float)
        raw_train_Y = label[indices[:split_num]].astype(float)
        raw_val_X = data[indices[split_num:]].astype(float)
        raw_val_Y = label[indices[split_num:]].astype(float)
        
        if USE_AUGMENTATION:
            aug_train_X, aug_train_Y = synthetic_interpolation(raw_train_X, raw_train_Y, factor=AUGMENTATION_FACTOR, include_origin=INCLUDE_ORIGIN)
        else:
            aug_train_X, aug_train_Y = raw_train_X, raw_train_Y
            
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        train_X = scaler_x.fit_transform(aug_train_X)
        val_X = scaler_x.transform(raw_val_X)
        train_Y = scaler_y.fit_transform(aug_train_Y)
        val_Y_scaled = scaler_y.transform(raw_val_Y)
        
        if test_data is not None:
            test_data_scaled = scaler_x.transform(test_data.astype(float))
        
        train_dataset = MSSDataset(train_X, train_Y)
        val_dataset = MSSDataset(val_X, val_Y_scaled)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Stats accumulators
        run_stats = {
            'min_loss': [], 'min_val_loss': [], 
            'min_mape': [], 'min_val_mape': [], 'avg_val_mape': []
        }
        
        best_model_wts = None
        best_val_loss = np.inf
        
        # Plotting
        fig,ax1,ax2=Make_Blank_Fig()

        # We'll generate the fig object inside loop but only save best run logic? 
        # Actually standard practice is plot per run or average. Let's stick to last run logic for fig, but stats average.
        
        for train_loop_idx in range(Train_repeats):
            if config_name == 'MLP':
                model = MSSModel(input_dim=inputN, output_dim=outputN, layer_sizes=ANN_upLayers, dropouts=ANN_upDrops).to(device)
            else:
                model = EnsembleNet(input_dim=inputN, output_dim=outputN, path1_dim=48*Node_N_ratio, path2_dim=96*Node_N_ratio).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
            criterion = nn.HuberLoss(delta=1.0)
            early_stopping = EarlyStopping(patience=Epoch_patient)
            
            history = {'loss': [], 'val_loss': [], 'mape': [], 'val_mape': []}
            
            for epoch in range(ANN_Epoch):
                model.train()
                r_loss, r_mape = 0.0, 0.0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    r_loss += loss.item() * bx.size(0)
                    r_mape += mean_absolute_percentage_error(by, out).item() * bx.size(0)
                
                hist_loss = r_loss / len(train_dataset)
                hist_mape = r_mape / len(train_dataset)
                
                model.eval()
                v_loss, v_mape = 0.0, 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        out = model(vx)
                        loss = criterion(out, vy)
                        v_loss += loss.item() * vx.size(0)
                        v_mape += mean_absolute_percentage_error(vy, out).item() * vx.size(0)
                
                hist_val_loss = v_loss / len(val_dataset)
                hist_val_mape = v_mape / len(val_dataset)
                
                scheduler.step(hist_val_loss)
                
                history['loss'].append(hist_loss)
                history['mape'].append(hist_mape)
                history['val_loss'].append(hist_val_loss)
                history['val_mape'].append(hist_val_mape)
                
                early_stopping(hist_val_loss, model)
                if early_stopping.early_stop:
                    early_stopping.restore_best_weights(model)
                    break
            
            # End of Epoch Loop
            min_v = min(history['val_loss'])
            if min_v < best_val_loss:
                best_val_loss = min_v
                best_model_wts = copy.deepcopy(model.state_dict())
            
            run_stats['min_loss'].append(min(history['loss']))
            run_stats['min_val_loss'].append(min(history['val_loss']))
            run_stats['min_mape'].append(min(history['mape']))
            run_stats['min_val_mape'].append(min(history['val_mape'])) 
            run_stats['avg_val_mape'].append(np.mean(history['val_mape'][-15:]))
            
            Plot_Trained_Fig(history, config_name, combo_names_str, ax1, ax2)
            print(f"    Repeat {train_loop_idx}: Val MAPE {min(history['val_mape']):.3f}")

        # Avg Stats
        avg_stats = {k: np.mean(v) for k, v in run_stats.items()}
        std_stats = {f"std_{k}": np.std(v) for k, v in run_stats.items()}
        
        # Save Fig
        Save_Trained_Fig(fig, config_name, combo_id_str)
        
        # Save Model
        save_packet = {
            'model_state_dict': best_model_wts,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'input_dim': inputN,
            'output_dim': outputN,
            'config': config_name,
            'combo_indices': combo_indices
        }
        with open(root_path + f'model_{config_name}_{combo_id_str}.pkl', 'wb') as f:
            pickle.dump(save_packet, f)
        
        # Save Aug Data
        if USE_AUGMENTATION:
            empty_col = np.full((aug_train_X.shape[0], 1), np.nan)
            aug_combined = np.hstack([aug_train_X, empty_col, aug_train_Y])
            df_aug = pd.DataFrame(aug_combined)
            aug_save_path = root_path + f'dataset_{config_name}_aug_{combo_id_str}.xlsx'
            with pd.ExcelWriter(aug_save_path) as writer:
                df_aug.to_excel(writer, sheet_name='aug_data', header=False, index=False)
        
        # Predict
        pred_array = None
        if test_data is not None:
             final_model = model # Re-instantiate if needed or reuse structure
             final_model.load_state_dict(best_model_wts)
             final_model.eval()
             with torch.no_grad():
                 t_in = torch.FloatTensor(test_data_scaled).to(device)
                 p_out = final_model(t_in).cpu().numpy()
                 pred_array = scaler_y.inverse_transform(p_out)

        # Record Result Entry
        entry = {
            'name': combo_names_str,
            'id': combo_id_str,
            'indices': combo_indices,
            'pred': pred_array,
            **avg_stats,
            **std_stats
        }
        results_store[config_name].append(entry)


# "Feature Presence Heatmap" with Ablation Logic

def plot_feature_presence_heatmap(results_list, config_name, top_n=30):
    # Mapping for easy lookup: tuple(indices) -> val_mape
    # Indices must be sorted tuple for consistency
    results_map = {tuple(sorted(res['indices'])): res['min_val_mape'] for res in results_list}
    
    # Sort by val_mape ascending (lower is better)
    sorted_res = sorted(results_list, key=lambda x: x['min_val_mape'])
    
    # Take top N
    if len(sorted_res) > top_n:
        sorted_res = sorted_res[:top_n]
    
    # Matrix: Rows = Top Models, Cols = Sheets
    # Value = Delta MAPE (MAPE_without_feature - MAPE_with_feature)
    # Positive Value = Feature Helped (Removing it made error worse)
    # 0 = Feature Not Present
    # NaN/Zero for single feature models where removal = unknown
    
    matrix = np.zeros((len(sorted_res), ScanN))
    model_labels = []
    scores = []
    
    for i, res in enumerate(sorted_res):
        current_indices = set(res['indices'])
        current_mape = res['min_val_mape']
        
        for idx in current_indices:
            # Sub-model without this feature
            sub_indices = tuple(sorted(list(current_indices - {idx})))
            
            if sub_indices in results_map:
                sub_mape = results_map[sub_indices]
                # Delta: (Error Without) - (Error With)
                # If Error Without (7) > Error With (5), Delta = +2 (Feature Helped)
                delta = sub_mape - current_mape
                matrix[i, idx] = delta
            else:
                # Sub-model not found (e.g., removing from single feature model -> empty)
                # Mark as 0 or a specific small value to indicate presence but undefined impact
                # Let's use 0 ensures it's neutral, but might look like "not present".
                # User asked for "increase or decrease", implying impact.
                # If unique feature, impact is undefined relative to subset.
                matrix[i, idx] = 0 
                
        model_labels.append(res['name'])
        scores.append(res['min_val_mape'])
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, len(sorted_res)*0.7), gridspec_kw={'width_ratios': [3, 1]})
    
    # Feature Heatmap (Impact)
    # Center colormap at 0. Red (Negative/Bad) < 0 < Blue (Positive/Good)
    # Present but 0 impact (or single feature) will be white-ish.
    # Non-present 0 will also be white-ish. 
    # To distinguish "Not Present" from "0 Impact", we could mask 0s?
    # But usually 0 impact is rare. Let's trust RdBu.
    
    max_val = np.max(np.abs(matrix)) if np.max(np.abs(matrix)) > 0 else 1
    sns.heatmap(matrix, ax=ax1, cmap="PiYG", center=0, cbar=True, vmin=-max_val, vmax=max_val,
                linewidths=0.5, linecolor='gray', annot=True, fmt=".1f")
                
    ax1.set_xticks(np.arange(ScanN) + 0.5)
    ax1.set_xticklabels(sheet_names_list, rotation=45, ha='right')
    ax1.set_yticks(np.arange(len(sorted_res)) + 0.5)
    ax1.set_yticklabels(model_labels, rotation=0)
    ax1.set_title(f"Top {len(sorted_res)} Models - Feature Ablation Impact ({config_name})\n(Positive = Feature Reduced Error)")
    
    # Score Heatmap (Single Col)
    score_matrix = np.array(scores).reshape(-1, 1)
    # Use different cmap for scores (lower is better, so maybe reversed?)
    sns.heatmap(score_matrix, ax=ax2, cmap="RdBu", annot=True, fmt=".2f", cbar=True, yticklabels=False)
    ax2.set_yticks(np.arange(len(sorted_res)) + 0.5)
    ax2.set_yticklabels(model_labels, rotation=0)
    ax2.set_title("Val MAPE")
    
    plt.tight_layout()
    plt.savefig(root_path + f'summary_heatmap_{config_name}.pdf')
    plt.close()


for config_name in configs:
    rs_list = results_store[config_name]
    
    # excel output
    df_res = pd.DataFrame(rs_list)
    # Remove large objects (pred) for summary sheet
    df_summary = df_res.drop(columns=['pred'])
    df_summary = df_summary.sort_values(by='min_val_mape')
    
    with pd.ExcelWriter(root_path + f'summary_{config_name}.xlsx') as writer:
        df_summary.to_excel(writer, sheet_name='Ranking', index=False)
        
        # Save Preds
        for res in rs_list:
            if res['pred'] is not None:
                short_name = res['id'] # use ID to keep sheet name short
                pd.DataFrame(res['pred']).to_excel(writer, sheet_name=f'pred_{short_name}')
                
    # Plot
    plot_feature_presence_heatmap(rs_list, config_name, top_n=12) # Show top n


print("All Done (v3.0).")
