import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from train.train import train_evaluate, evaluate
from utils import EmbedEncode, LoadBalancingLoss
from dataset.darkphotondataset import DarkPhotonDataset
from dataset.transforms import GraphFilter
from models.model import Transformer
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

input_size = 4  # fixed
hidden_size = 80  # hidden size of the modules
encoding_size = 4  # size of the positional encoding
g_norm = False  # normalized or unnormalized data
heads = 2  # number of attention heads
num_xprtz = 6  # number of experts of the moe layer
xprt_size = 30  # hidden size of each expert
k = 2  # top k experts to activate at a time
dropout_encoder = 0.2  # dropout probability
layers = 2  # number of endoder blocks in the transformer
output_size = 2  # fixed
w_load = 1  # weight of the load balancing loss
batchsize = 256
epochs = 80
patience = 45  # patience of the stepper
gamma = 0.1  # multiplicative factor for the stepper
learning_rate = 0.002
stepper = True
con = str(str(input_size)+str(hidden_size)+str(encoding_size)+str(g_norm)+str(heads)+str(num_xprtz)+str(xprt_size)+str(k)+str(dropout_encoder)+str(layers)+str(output_size)+str(w_load)+str(batchsize)+str(epochs)+str(patience)+str(gamma)+str(learning_rate)+str(stepper)+str("prova"))

save_path = f'./results/{con}'
os.makedirs(save_path, exist_ok=True)

# ---- SET RANDOM SEEDS ----
seed = 3958239256
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
graph_filter = GraphFilter(min_num_nodes=2) # only graphs with at least 2 nodes will be accepted
dataset = DarkPhotonDataset(root=dataset_root, 
                            subset=1.0,
                            url="https://cernbox.cern.ch/s/PYurUUzcNdXEGpz/download",
                            pre_filter = graph_filter,
                            pre_transform = None,
                            post_filter = None, 
                            verbose=True)

trainset, testset = train_test_split(dataset, test_size=0.2)
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



