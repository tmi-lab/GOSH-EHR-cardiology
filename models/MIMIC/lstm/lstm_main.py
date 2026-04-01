import pandas as pd
import numpy as np
from datetime import datetime
import time
import transformers
import torch
import torch.nn as nn
import ast
import os
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, utils, BertModel, BertTokenizer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

import csv
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm.notebook import tqdm
import wandb
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from lstm_data import LSTMCustomDataset
from model import LSTMClassifier
from training import train_function
from evaluation import bootstrap_test_evaluation
from torch.utils.data import Dataset, DataLoader, TensorDataset
#from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel



print('LOADING DATA')
train = torch.load('Path/to/tokenised/ROBbert_train_lls_tokenised_binary_mimic.pth', weights_only = False) # <-- Update the path to your tokenised training data
val = torch.load('Path/to/tokenised/ROBbert_val_lls_tokenised_binary_mimic.pth', weights_only = False) # <-- Update the path to your tokenised validation data

print('DATA LOADED')

batch_size = 16 

train_dataset = LSTMCustomDataset(train)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = LSTMCustomDataset(val)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

vocab_size = tokenizer.vocab_size
embed_dim = 128
hidden_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('LOAD MODEL')
model = LSTMClassifier(vocab_size, embed_dim, hidden_dim)
model.to(device)

optimizer = AdamW(model.parameters(), lr = 2e-6, eps=1e-8)

epochs = 50
loss_function = nn.CrossEntropyLoss()

total_steps = len(train_data_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
print('START TRAINING')

csv_file_path = "Path/to/training_results_LSTM_binary_mimic.csv" # <-- Update the path to save your training results, e.g., "lstm/training_results_LSTM_binary_mimic.csv"
                  
with open(csv_file_path, "w", newline="") as csvfile:
    model = train_function(
        model, optimizer, scheduler, epochs, loss_function,
        train_data_loader, val_data_loader, device,
        clip_value=2.0,
        csv_file=csvfile,
        patience=5,           # x epochs
        min_delta=1e-4,        # optional threshold
        use_wandb=True,
        wandb_project="Project Name", # <-- Update with your W&B project name
        wandb_run_name="LSTM_mimic_los",
        wandb_config={
            "epochs": epochs,
            "clip_value": 2.0,
            "patience": 5,
            "min_delta": 1e-4,
        },
    )
                  
print('LOAD TEST')
                
test = torch.load('Path/to/tokenised/ROBbert_test_lls_tokenised_binary_mimic.pth', weights_only = False) # <-- Update the path to your tokenised test data, e.g., "lstm/ROBbert_test_lls_tokenised_binary_mimic.pth"
test_dataset = LSTMCustomDataset(test)

test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('BOOTSTRAP')
summary_df = bootstrap_test_evaluation(
    model=model,
    test_data_loader=test_data_loader,
    loss_function=torch.nn.CrossEntropyLoss(),
    device=device,
    n_bootstrap=1000,
    csv_path="Path/to/lstm_bootstrap_summary.csv" # <-- Update the path to save your bootstrap summary results
)
                  
print('done!')