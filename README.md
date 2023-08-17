# Optimizing Robot Social Navigation System using Continuous Action Spaces



# LSTM Based Sequential Data Prediction

This repository contains a set of Python scripts that illustrate how to train a Long Short-Term Memory (LSTM) network for sequential data prediction. The example provided here processes input data, trains an LSTM network, and evaluates its performance on validation data. The implementation is based on the PyTorch library.

## 🛠️ Requirements

- Python 3.x
- PyTorch
- numpy
- TensorBoard (for visualization)

## 🚀 Quick Start

### 1. **Data Preprocessing**
   - Run the `process_data.py` script.
     ```bash
     python process_data.py
     ```
   - It processes data into input-output pairs saved in `.npz` format in the `processed_data` directory.

### 2. **Train the LSTM Model**
   - Run the `train_rnn.py` script.
     ```bash
     python train_rnn.py
     ```
   - Model checkpoints and TensorBoard logs will be saved in the `ckpt` and `runs` directories, respectively.


### 3. **Visualizing Training Progress**
   - Run TensorBoard by executing:
     ```bash
     tensorboard --logdir=runs
     ```
   - Navigate to `http://localhost:6006/` to view the training progress.



