import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric's DataLoader
import numpy as np
import random
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torchmetrics.classification import AUROC  # don't forget to import this if using torchmetrics

from train import train_evaluate, evaluate
from utils import EmbedEncode, LoadBalancingLoss
from dataset import CustomEventsDataset2, MakeHomogeneous
from model import Transformer
from sklearn.model_selection import train_test_split

# ---- SETUP for evaluation ----
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_auroc_torchmetrics = []
all_auroc_sklearn = []

device = torch.device('cuda:0') #if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---- CONFIGURATION ----
encoding_size = 4
trials = 10

input_size = 12
hidden_size = 80
g_norm = True
heads = 2
num_xprtz = 6
xprt_size = 20
k = 2
dropout_encoder = 0.0
layers = 2
output_size = 2
w_load = 1
batchsize = 500
epochs = 60
patience = 45
gamma = 0.1
learning_rate = 0.001
stepper = True
con = str(str(input_size)+str(hidden_size)+str(encoding_size)+str(g_norm)+str(heads)+str(num_xprtz)+str(xprt_size)+str(k)+str(dropout_encoder)+str(layers)+str(output_size)+str(w_load)+str(batchsize)+str(epochs)+str(patience)+str(gamma)+str(learning_rate)+str(stepper)+str("prova"))

save_path = f'./results/{con}'
os.makedirs(save_path, exist_ok=True)

# ---- SET RANDOM SEEDS ----
seed = 42
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_root = './data'
os.makedirs(dataset_root, exist_ok=True)

# ---- LOAD DATA ----
datasetfull = CustomEventsDataset2(
    root=dataset_root,
    url='https://cernbox.cern.ch/s/0nh0g7VubM4ndoh/download',
    k=encoding_size,
    delete_raw_archive=False,
    add_edge_index=True,
    event_subsets={'signal': 400, 'singletop': 200, 'ttbar': 200},
    transform=MakeHomogeneous(), 
    signal_filter=lambda filename: "Wh_hbb_fullMix.h5" in filename
)

trainset, testset = train_test_split(datasetfull, test_size=0.2)
trainset, evalset = train_test_split(trainset, test_size=0.2)

# ---- BUILD MODEL ----
model = Transformer(input_size, hidden_size, encoding_size, g_norm, heads, num_xprtz, xprt_size, k, dropout_encoder, layers, output_size).to(device)

if learning_rate is not None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = optim.Adam(model.parameters())

if stepper:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=gamma)
else:
    scheduler = None

criterion = nn.CrossEntropyLoss()
loss_fn = LoadBalancingLoss(criterion, w_load)

# ---- DATALOADERS ----
# ---- DATALOADERS ----
train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
eval_loader = DataLoader(evalset, batch_size=batchsize, shuffle=False)
test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False)

# ---- TRAIN ----
result1 = train_evaluate(train_loader, eval_loader, model, criterion, loss_fn, optimizer, scheduler, patience, epochs, device = device)

# ---- TEST ----
test_loader = DataLoader(testset, shuffle=False, batch_size=batchsize)
result2 = evaluate(test_loader, model, loss_fn, device = device)

# ---- METRICS ----
predictions = torch.argmax(torch.cat(result2[0]), dim=1)
truths = torch.cat(result2[1])

confmat = confusion_matrix(truths.cpu(), predictions.cpu())
test_acc = accuracy_score(truths.cpu(), predictions.cpu())
test_precision = precision_score(truths.cpu(), predictions.cpu())
test_recall = recall_score(truths.cpu(), predictions.cpu())
test_f1 = f1_score(truths.cpu(), predictions.cpu())

# AUROC using torchmetrics
auroc_torchmetrics = AUROC(task='binary')
test_auroc_torchmetrics = auroc_torchmetrics(torch.cat(result2[0])[:, 1], truths)

# AUROC using sklearn
test_auroc_sklearn = roc_auc_score(truths.cpu(), torch.cat(result2[0])[:, 1].cpu())

# Save metrics to lists
all_accuracy.append(test_acc)
all_precision.append(test_precision)
all_recall.append(test_recall)
all_f1.append(test_f1)
all_auroc_torchmetrics.append(test_auroc_torchmetrics.item())
all_auroc_sklearn.append(test_auroc_sklearn)

# ---- SAVE RESULTS ----
array_test_df = {
    "config": con,
    "confusion matrix": confmat.tolist(),  # to make it JSON serializable if needed
    "accuracy": test_acc,
    "precision": test_precision,
    "recall": test_recall,
    "f1_score": test_f1,
    "auroc_torchmetrics": test_auroc_torchmetrics.item(),
    "auroc_sklearn": test_auroc_sklearn
}
test_df = pd.DataFrame([array_test_df])

# Save DataFrame as CSV
test_df.to_csv(os.path.join(save_path, 'test_metrics.csv'), index=False)

# Save model
torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pt'))
print(f"Training complete. Model and metrics saved in '{save_path}'")

