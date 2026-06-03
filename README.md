# Source Code Architecture (`/src`)

This directory contains the core modularized implementation of the Spatio-Temporal Multi-Task Learning pipeline. The architecture strictly decouples data processing, model definition, training loops, and evaluation metrics to ensure high maintainability and scalability.

## Module Breakdown

### 1. `models.py`
Defines the neural network architectures utilizing PyTorch[cite: 18]. 
* **`MultiTaskGRU` & `AttentionGRU`:** The primary models featuring a shared temporal encoder with dual task-specific heads[cite: 18].
  * *Regression Head:* Predicts displacement across multiple future horizons (`h1`, `h2`, `h3`, `cum_disp`)[cite: 18].
  * *Classification Head:* Outputs categorical risk levels[cite: 18].
* **Ablation Models:** Includes `LSTMModel` and `TCNModel` (Temporal Convolutional Network) for baseline comparisons[cite: 18].

### 2. `train.py`
Contains the custom `Trainer` class[cite: 20].
* Implements the joint loss function: `Loss = MSELoss + λ * CrossEntropyLoss`[cite: 20].
* Prevents exploding gradients using `torch.nn.utils.clip_grad_norm_`[cite: 20].
* Integrates `ReduceLROnPlateau` for adaptive learning rate scheduling based on validation loss[cite: 20].

### 3. `evaluate.py`
The comprehensive `Evaluator` class for model diagnostics[cite: 17].
* Calculates multi-dimensional metrics (MAE, RMSE, R², Accuracy, F1-score)[cite: 17].
* Performs **Zone-Based Evaluation** (`evaluate_by_zone`) to analyze performance discrepancies across different geological deformation zones[cite: 17].
* Automatically generates diagnostic plots (Prediction Curves, Error Distributions, Confusion Matrices)[cite: 17].

### 4. `data_loader.py` & `preprocess.py`
Handles the ingestion and transformation of heterogeneous multi-source data[cite: 16, 19].
* **`DataLoader`:** Parses node info and manages datetime formatting alignment[cite: 16].
* **`DataPreprocessor`:** Enforces strict chronological data splitting (`split_by_time`) to prevent temporal data leakage and handles standard scaling exclusively fitted on the training distribution[cite: 19].

### 5. `utils.py`
Utility functions ensuring reproducibility and I/O management[cite: 15].
* `set_seed()`: Enforces deterministic behavior across numpy and CUDA backends[cite: 15].
* `save_metrics()` & `save_predictions()`: Exports evaluation artifacts to JSON and CSV formats[cite: 15].
