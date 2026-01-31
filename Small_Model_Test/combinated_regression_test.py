import os
import sys
import pandas as pd
import numpy as np
import itertools
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

import subprocess
import json
import shutil

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class CombinatedAETester:
    def __init__(self, file_path, save_parent_dir='combinated_ae_results'):
        self.file_path = file_path
        self.save_parent_dir = save_parent_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(save_parent_dir, f'run_{self.timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.sheets = {}
        self.sheet_names = []
        self.load_sheets()
        
        self.results_summary = []

    def load_sheets(self):
        """讀取Excel的所有分頁"""
        print(f"正在讀取 Excel 檔案: {self.file_path}")
        try:
            xls = pd.ExcelFile(self.file_path)
            self.sheet_names = xls.sheet_names
            print(f"檢測到 {len(self.sheet_names)} 個分頁: {self.sheet_names}")
            
            for name in self.sheet_names:
                df = pd.read_excel(xls, sheet_name=name)
                print(f"  - 分頁 '{name}' 形狀: {df.shape}")
                self.sheets[name] = df
                
        except Exception as e:
            print(f"讀取檔案失敗: {e}")
            sys.exit(1)
        
    def parse_sheet_columns(self, df):
        """解析分頁的輸入和輸出列 (參考 SpectraAETrainer 的邏輯)"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return [], []
            
        # 關鍵字檢測
        input_candidates = [c for c in numeric_cols if any(k in c.upper() for k in ['IN', 'X', 'SOURCE', 'RAW'])]
        target_candidates = [c for c in numeric_cols if any(k in c.upper() for k in ['OUT', 'Y', 'TARGET', 'REF'])]
        
        if input_candidates and target_candidates:
            return input_candidates, target_candidates
        
        # 空白列分隔檢測
        split_index = -1
        for i, col in enumerate(numeric_cols):
            if df[col].isna().all():
                split_index = i
                break
        
        if split_index != -1:
            input_cols = numeric_cols[:split_index]
            target_cols = numeric_cols[split_index+1:]
            return input_cols, target_cols
            
        # 默認分割 (60/40)
        n_cols = len(numeric_cols)
        split_idx = int(n_cols * 0.6)
        input_cols = numeric_cols[:split_idx]
        target_cols = numeric_cols[split_idx:]
        return input_cols, target_cols

    def merge_data(self, combo_names):
        """合併指定的資料表 (先合併所有 Input, 再合併所有 Output)"""
        if not combo_names:
            return None
            
        base_df = self.sheets[combo_names[0]]
        n_rows = len(base_df)
        
        # 檢查行數是否一致
        for name in combo_names[1:]:
            if len(self.sheets[name]) != n_rows:
                print(f"警告: 分頁 '{name}' ({len(self.sheets[name])} 行) 與 '{combo_names[0]}' ({n_rows} 行) 行數不一致，跳過此組合。")
                return None
        
        all_inputs = []
        all_targets = []
        
        for name in combo_names:
            df = self.sheets[name].copy()
            df = df.reset_index(drop=True) # 確保索引重置
            
            # 解析列
            in_cols, out_cols = self.parse_sheet_columns(df)
            
            # 提取並重命名
            df_in = df[in_cols].copy()
            df_in.columns = [f"{name}_{c}" for c in df_in.columns]
            all_inputs.append(df_in)
            
            df_out = df[out_cols].copy()
            df_out.columns = [f"{name}_{c}" for c in df_out.columns]
            all_targets.append(df_out)
            
        # 合併所有的 Input
        merged_inputs = pd.concat(all_inputs, axis=1)
        # 合併所有的 Output
        merged_targets = pd.concat([all_targets[0]], axis=1)  # 特殊情況!! 目前output都是一樣的，所以只取第一個 Output
        
        # 建立空白分隔列
        blank_col = pd.DataFrame(np.nan, index=merged_inputs.index, columns=['SEPARATOR'])
        
        # 最終合併 (Input在前, 空白列, Output在後)
        merged_df = pd.concat([merged_inputs, blank_col, merged_targets], axis=1)
        return merged_df

    def run(self, aug_factor=6, aug_include_ori=False, epochs=500, patience=50):
        """執行所有排列組合的訓練"""
        
        # 生成所有組合 (1 ~ N 個分頁)
        all_combos = []
        for r in range(1, len(self.sheet_names) + 1):
            combos = list(itertools.combinations(self.sheet_names, r))
            all_combos.extend(combos)
            
        print(f"\n總共將執行 {len(all_combos)} 種分頁組合測試。")
        print("="*60)
        
        # 定位 pytorch_regression_system.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_script = os.path.join(script_dir, 'pytorch_regression_system.py')
        
        if not os.path.exists(target_script):
            print(f"錯誤: 找不到 {target_script}")
            return

        for i, combo in enumerate(all_combos, 1):
            combo_name = "+".join(combo)
            print(f"\n[組合 {i}/{len(all_combos)}]: {combo_name}")
            
            # 1. 合併資料
            merged_df = self.merge_data(combo)
            if merged_df is None:
                continue
                
            # 2. 保存合併後的檔案
            combo_dir = os.path.join(self.run_dir, combo_name)
            os.makedirs(combo_dir, exist_ok=True)
            data_save_path = os.path.join(combo_dir, 'merged_data.xlsx')
            merged_df.to_excel(data_save_path, index=False)
            
            # 3. 調用 pytorch_regression_system.py (CLI)
            training_results_dir = os.path.join(combo_dir, 'training_results')
            os.makedirs(training_results_dir, exist_ok=True)
            
            cmd = [
                sys.executable,
                '-u',  # Force unbuffered output
                target_script,
                '--file', data_save_path,
                '--output_dir', training_results_dir,
                '--aug_count', str(aug_factor),
                '--patience', str(patience),
                #'--no_save_report', # 不讓子程序生成報告，我們自己生成總報告
                #'--no_save_model',  # 不存模型以節省空間，或者留著? 原代碼有存。這裡先不存
                '--save_data'       # 存下每個組合的增強數據? 根據需求。原需求說 "自動儲存每個merge的結果來進行比較"，指的可能是metrics
            ]
            
            if aug_include_ori:
                cmd.append('--include_original')
                
            print(f"  執行 CLI: {' '.join(cmd)}")
            
            # Force UTF-8 encoding for the subprocess
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            try:
                # 執行子程序 (改為實時輸出)
                process = subprocess.Popen(
                    cmd, 
                    # close_fds=True, # Windows 上通常不需要，且可能導致 pipe 問題，先註解掉
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True, 
                    encoding='utf-8', 
                    errors='replace',     # Handle encoding errors gracefully
                    bufsize=1,
                    universal_newlines=True,
                    env=env               # Pass the environment with forced encoding
                )
                
                # 實時讀取並顯示輸出
                stdout_lines = []
                while True:
                    line = process.stdout.readline()
                    # print(line) # Debug print removed
                    if not line and process.poll() is not None:
                        break
                    if line:
                        print(f"    | {line.strip()}") 
                        stdout_lines.append(line)
                        
                # 等待程序結束並獲取剩餘輸出
                try:
                    stdout_final, _ = process.communicate(timeout=5) # 應該已經讀完了，加個 timeout 保險
                except:
                    stdout_final = ""
                
                if stdout_final:
                    for line in stdout_final.splitlines():
                        print(f"    | {line.strip()}")
                        stdout_lines.append(line)
                
                if process.returncode != 0:
                    print(f"  執行失敗 (Code {process.returncode})")
                    # stderr 已經合併到 stdout了，所以上面已經印出來了
                    continue
                
                # 4. 讀取 results.json
                json_path = os.path.join(training_results_dir, 'results.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        res_data = json.load(f)
                        
                    best_model_name = res_data.get('best_model_name')
                    if best_model_name and best_model_name in res_data['results']:
                        best = res_data['results'][best_model_name]
                        data_info = res_data.get('data_info', {})
                        
                        self.results_summary.append({
                            'Combination': combo_name,
                            'Num_Sheets': len(combo),
                            'Input_Dim': data_info.get('n_features', 0),
                            'Output_Dim': data_info.get('n_targets', 0),
                            # best dict contains: model, scaler, augmentation, mae, mse, mape, r2...
                            'Best_Model': best['model_class'],
                            'Best_Scaler': best['scaler_name'],
                            'Best_Aug': best['augmentation'],
                            'MAPE': best.get('mape_mean', 0),
                            'MSE': best.get('rmse', 0)**2,
                            'MAE': best['mae'],
                            'R2': best['r2'],
                            'Actual_Epochs': best.get('stopped_epoch', 0)
                        })
                        print(f"  -> 完成: Best MAPE={best.get('mape_mean', 0):.2f}% ({best['model_class']})")
                    else:
                        print("  警告: resulst.json 中未找到最佳模型資訊")
                else:
                    print(f"  警告: 未找到結果檔案 {json_path}")
                    
            except Exception as e:
                print(f"組合 '{combo_name}' 執行時發生例外: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. 生成總結報告
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """生成所有組合的比較報告 (PDF)"""
        if not self.results_summary:
            print("沒有產生任何結果，無法生成報告。")
            return

        df_results = pd.DataFrame(self.results_summary)
        
        # 保存 CSV 摘要
        summary_csv_path = os.path.join(self.run_dir, 'combinations_summary.csv')
        df_results.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n摘要數據已保存至: {summary_csv_path}")
        
        pdf_path = os.path.join(self.run_dir, 'Combinations_Report.pdf')
        print(f"正在生成比較報告 PDF: {pdf_path} ...")
        
        with PdfPages(pdf_path) as pdf:
            # 頁面 1: 摘要列表
            plt.figure(figsize=(12, 10))
            plt.axis('off')
            plt.text(0.5, 0.95, f'Combinated AE Test Report', ha='center', va='top', fontsize=20, weight='bold')
            plt.text(0.5, 0.90, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ha='center', fontsize=12)
            
            # 取前 20 名展示 (如果太多)
            display_df = df_results.sort_values('MAPE').head(20)[['Combination', 'MAPE', 'R2', 'Best_Model']]
            pass # 表格繪製稍微複雜，這裡用簡單的文字列表或稍後用圖表呈現
            
            table_text = "Top Combinations (by MAPE):\n\n"
            table_text += f"{'Rank':<5} {'Combination':<30} {'MAPE':<10} {'R2':<10} {'Model':<15}\n"
            table_text += "-"*80 + "\n"
            
            for i, row in enumerate(display_df.itertuples(), 1):
                name = (row.Combination[:27] + '...') if len(row.Combination) > 27 else row.Combination
                table_text += f"{i:<5} {name:<30} {row.MAPE:6.2f}%    {row.R2:6.4f}     {row.Best_Model:<15}\n"
            
            plt.text(0.1, 0.80, table_text, family='monospace', va='top', fontsize=10)
            pdf.savefig()
            plt.close()
            
            # 頁面 2: MAPE 比較 (Bar Chart)
            plt.figure(figsize=(12, 8))
            # 按 MAPE 排序
            df_sorted = df_results.sort_values('MAPE', ascending=True) # 小的在上面
            sns.barplot(data=df_sorted, x='MAPE', y='Combination', palette='viridis')
            plt.title('MAPE by Combination (Lower is Better)')
            plt.xlabel('MAPE (%)')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 頁面 3: Metric Heatmap (如果有足夠多的數據點，這裡用 Correlation 或直接數值熱圖)
            # 這裡我們做一個 "Metrics Overview" 熱圖，標準化各指標以便在同一圖顯示，或者直接顯示數值
            # 為避免數值差異過大，我們分開畫或只畫 MAPE/R2
            
            metrics = ['MAPE', 'MAE', 'MSE', 'R2']
            plt.figure(figsize=(10, len(df_results) * 0.5 + 2))
            
            # 取出數值並設置索引
            heatmap_data = df_results.set_index('Combination')[metrics]
            
            # R2 越高越好，其他越低越好。為了視覺一致性，可以不處理，直接看顏色
            sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True)
            plt.title('Metrics Heatmap')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 頁面 4: 參數數量/維度 比較
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=df_results, x='Input_Dim', y='MAPE', hue='Num_Sheets', size='Num_Sheets', sizes=(50, 200))
            plt.title('Input Dimension vs MAPE')
            plt.xlabel('Input Dimension (Feature Count)')
            plt.ylabel('MAPE (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        # CLI Mode
        parser = argparse.ArgumentParser(description='Combinated AutoEncoder Test CLI')
        parser.add_argument('--file', type=str, required=True, help='Path to multi-sheet Excel file')
        parser.add_argument('--aug_factor', type=int, help='Augmentation factor')
        parser.add_argument('--include_original', action='store_true', help='Include original data in training set during augmentation')
        parser.add_argument('--epochs', type=int, help='Max training epochs')
        parser.add_argument('--patience', type=int, help='Early stopping patience')
        
        args = parser.parse_args()
        
        # 參數交互式詢問 (如果未提供)
        if args.aug_factor is None:
            val = input("請輸入數據增強倍數 (默認 6): ").strip()
            args.aug_factor = int(val) if val else 6
            
        if args.epochs is None:
            val = input("請輸入最大訓練 Epochs (默認 500): ").strip()
            args.epochs = int(val) if val else 500
            
        if args.patience is None:
            val = input("請輸入 Early Stopping Patience (默認 50): ").strip()
            args.patience = int(val) if val else 50
            
        if not args.include_original:
            if len(sys.argv) <= 3:
                 val = input("增強時是否包含原始數據? (y/N): ").strip().lower()
                 args.include_original = (val == 'y' or val == 'yes')

        tester = CombinatedAETester(file_path=args.file)
        tester.run(
            aug_factor=args.aug_factor,
            aug_include_ori=args.include_original,
            epochs=args.epochs,
            patience=args.patience
        )
    else:
        # Interactive Mode
        print("=" * 70)
        print("Combinated AutoEncoder Test - Interactive Mode")
        print("=" * 70)
        
        try:
            # 檔案選擇
            import tkinter as tk
            from tkinter import filedialog
            
            print("請選擇多頁 Excel 檔案...")
            root = tk.Tk()
            root.withdraw()
            
            file_path = filedialog.askopenfilename(
                title="選擇多頁 Excel 檔案",
                filetypes=[("Excel Files", "*.xlsx;*.xls")]
            )
            
            if not file_path:
                print("未選擇檔案，程序退出")
                sys.exit()
                
            print(f"已選擇檔案: {file_path}")
            
            # 詢問參數
            print("-" * 60)
            
            # Aug Factor
            val = input("請輸入數據增強倍數 (默認 6): ").strip()
            aug_factor = int(val) if val else 6
            
            # Include Original
            val = input("增強時是否包含原始數據? (y/N): ").strip().lower()
            aug_include_ori = (val == 'y' or val == 'yes')
            
            # Epochs
            val = input("請輸入最大訓練 Epochs (默認 500): ").strip()
            epochs = int(val) if val else 500
            
            # Patience
            val = input("請輸入 Early Stopping Patience (默認 50): ").strip()
            patience = int(val) if val else 50
            
            print(f"設定: 增強={aug_factor}x, 包含原始={aug_include_ori}, Epochs={epochs}, Patience={patience}")
            
            tester = CombinatedAETester(file_path=file_path)
            tester.run(
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
        print("=" * 70)