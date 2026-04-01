# MIMIC – Length of Stay Prediction

This folder contains experiments for predicting hospital **Length of Stay (LoS)** using MIMIC electronic health record data. It includes both notebook-based exploration and a structured LSTM pipeline.

## Contents

### Jupyter Notebooks

- `MIMIC_BioClinicalBERT_LoS.ipynb`  
  End-to-end workflow for train-val-splits needed in all models, training and evaluation of fine-tuned BioClinicalBERT for Length of Stay task. 

- `roberta_los.ipynb`  
  Transformer-based approach (RoBERTa) for LoS prediction using textual representations.

- `LoS_lstm.ipynb`  
  Additional evaluation steps for Length of Stay LSTM model (requires best_lstm_model.pth).

---

### lstm/

Modular implementation of the LSTM pipeline:

- `lstm_main.py` – Main script to run training and evaluation  
- `model.py` – LSTM model architecture  
- `training.py` – Training loop and optimization  
- `lstm_data.py` – Data loading and preprocessing  
- `evaluation.py` – Model evaluation and metrics  

---

## Setup

- Paths to datasets and saved models **must be updated locally**.  
- All required path changes are **clearly indicated in the code** (e.g., comments such as `# UPDATE PATH HERE`).  
- Install required dependencies before running (e.g., `torch`, `pandas`, `numpy`).

---

## Usage

Run the LSTM pipeline:

```bash
python lstm/lstm_main.py