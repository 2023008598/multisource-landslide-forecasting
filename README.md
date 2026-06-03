# Multi-Source Landslide Forecasting & Joint Warning System

This repository implements an advanced time-series forecasting and multi-task learning pipeline for landslide displacement prediction and risk level early warning, utilizing heterogeneous geological monitoring data.

## Project Overview
Predicting geological hazards requires fusing multi-source data (Rainfall, Water Level, GPS, InSAR, etc.) with different sampling frequencies. This project constructs a "Driving Factor Extraction + Temporal Modeling + Spatial Constraint + Dual-Objective Output" framework.

It simultaneously outputs:
1. **Displacement Prediction (Regression):** Continuous monitoring point displacement sequences over a future horizon (`HORIZON = 3` months).
2. **Risk Evolution (Classification):** Categorical risk levels (Blue, Yellow, Orange, Red) triggered within the exact same prediction window.

## Core Architecture: Multi-Task Attention-GRU
Instead of training isolated models, the core network utilizes an `AttentionGRU` architecture that shares a temporal encoder to capture the latent dynamics of delayed rainfall and water-level fluctuations. 

**Advanced Engineering Features:**
* **Joint Multi-Task Loss:** Optimizes a dynamic objective function balancing regression and classification:
  Loss = MSE(Displacement) + λ × CrossEntropy(Risk)
  *(Default λ = 0.5)*
* **Gradient Clipping & Adaptive LR:** Implements `clip_grad_norm_` to prevent exploding gradients in complex RNN structures, and uses `ReduceLROnPlateau` for precise convergence tracking.
* **Ablation Models Available:** The architecture module provides standard `MultiTaskGRU`, `LSTMModel`, and `TCNModel` for robust baseline benchmarking and ablation studies.

## Deep Dive: Robustness & Data Leakage Prevention

**1. Strict Prevention of Temporal Leakage**
A common pitfall in time-series deep learning is randomized data splitting, which inadvertently leaks future information. This pipeline enforces a strict chronological split:
* **Train Set:** Up to `2023-12`
* **Validation Set:** `2024-01` to `2024-06`
* **Test Set:** `2024-07` onwards
Data standardizations and threshold calibrations are strictly computed *only* on the training distribution.

**2. Comprehensive Evaluation & Zone Analysis**
The customized `Evaluator` class goes beyond simple metrics. It evaluates the model across different spatial deformation zones and generates diagnostic artifacts including:
* Prediction curves across distinct GPS nodes.
* Error distribution histograms.
* Multiclass Confusion Matrices for risk levels.
* Spatial Feature Importance analysis.

## Repository Structure
* `config.py`: Centralized configuration for hyperparameters, data paths, and dynamic risk thresholds.
* `main.py`: Entry point orchestrating the end-to-end multi-task execution pipeline (comparing Baseline vs. Improved models).
* `/src`: Highly modularized source code:
  * `data_loader.py` & `preprocess.py`: Handles complex temporal alignment and NaN processing.
  * `models.py`: Contains all neural architectures (`AttentionGRU`, `MultiTaskGRU`, `TCN`, etc.).
  * `train.py`: Custom PyTorch Trainer logic with multi-task loss management.
  * `evaluate.py`: Generates rigorous metrics (MAE, RMSE, R², F1) and analytical plots.
* `/outputs`: Centralized storage for training checkpoints (`/models`), inference CSVs and JSON metrics (`/results`), and generated diagnostic figures (`/figures`).
