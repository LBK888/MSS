# MSS Deep Learning Project

This repository contains a suite of tools designed for deep learning research, focusing on small dataset regression and multispectral spectrum conversion (AutoEncoder). It includes tools for finding the best model architectures, scalers, and augmentation strategies, as well as a specific implementation for White Shrimp MSS (Multi-Spectral Sensor) data analysis.

## Components

### 1. Small Model Test
**Location:** `Small_Model_Test/`

A powerful tool designed to extensively test and identify the optimal model architecture, data scaler, and augmentation algorithms for **small datasets**.

*   **Features:**
    *   **Automated Grid Search:** Automatically tests combinations of various architectures, scalers, and augmentation methods.
    *   **Supported Architectures:** Deep Feedforward Networks (DeepFFN), Wide Networks, ResNet, EnsembleNet, and AutoEncoderNet.
    *   **Data Augmentation:** Varied techniques including Gaussian Noise, Feature Scaling, Mixup, Bootstrap, Feature Dropout, SMOTE-style, and Synthetic Interpolation.
    *   **Scalers:** StandardScaler, MinMaxScaler, RobustScaler.
    *   **Reporting:** Generates detailed markdown reports and visualizations of top-performing configurations.
    *   **CLI Support:** Run experiments easily via command line arguments.

### 2. Spectra AE (AutoEncoder) System
**Location:** `Spectra_AE/`

A specialized tool for optimizing **Multispectral Wavelength Conversion** tasks. It systematically searches for the best AutoEncoder model architecture, scaler, and augmentation algorithms.

*   **Features:**
    *   **Architectures:** Includes SimpleAE, DeepAE (with BatchNorm/Dropout), ResidualAE (ResNet-style), WideAE, and BottleneckAE.
    *   **Augmentation:** Domain-specific augmentations such as Intensity Scaling, Baseline Shift, Gaussian Noise, and Mixup.
    *   **Benchmarking:** Runs 120+ combinations (Model × Scaler × Augmentation) to find the global optimum.
    *   **Output:** Automatically saves the top 3 performing models with ready-to-use Python inference code and detailed performance metrics (MSE, MAE, RMSE, R², MAPE).

### 3. White Shrimp MSS PyTorch System
**Filename:** `white_shrimp_mss_pytorch_v3.py`

The core application that leverages the insights gained from the *Small Model Test* and *Spectra AE* tools. It performs combinatorial analysis on multiple small datasets, training deep learning regression models to predict specific outcomes for white shrimp.

*   **Purpose:** To find the best regression deep learning model by performing multiple training runs on permutations of datasets using optimized architectures found by the previous tools.
*   **Features:**
    *   **PyTorch Implementation:** High-performance training with CUDA support.
    *   **Advanced Training:** Implements Early Stopping, Learning Rate Scheduling (ReduceLROnPlateau), and Ensemble methods.
    *   **Data Handling:** Intelligent handling of small datasets with synthetic interpolation augmentation.
    *   **Visualization:** Generates comprehensive plots for training history, prediction accuracy (MAPE), and feature importance.

## Requirements

*   Python 3.8+
*   PyTorch
*   Pandas, NumPy, Scikit-learn
*   Matplotlib, Seaborn

## License

This project is licensed under the Apache-2.0 License.
