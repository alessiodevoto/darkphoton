# evaluate_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import random
import re
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

from dataset.darkphotondataset import DarkPhotonDataset
from dataset.transforms import GraphFilter
from models.model import Transformer
from utils import LoadBalancingLoss
from train.train import evaluate

# ---- CONFIGURATION ----
input_size = 4
hidden_size = 80
encoding_size = 4
g_norm = False
heads = 2
num_xprtz = 6
xprt_size = 30
k = 2
dropout_encoder = 0.2
layers = 2
output_size = 2
w_load = 1
batchsize = 256
seed = 3958239256
model_ckpt_path = '/hdd3/dongen/Desktop/DarkPhoton/darkphoton/results/4804False263020.222125680450.10.002Trueprova/final_model.pt'
output_dir = './evaluation_outputs'

os.makedirs(output_dir, exist_ok=True)

# ---- SET SEEDS ----
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_root = './data'
os.makedirs(dataset_root, exist_ok=True)

# ---- LOAD DATA ----
graph_filter = GraphFilter(min_num_nodes=2)
dataset = DarkPhotonDataset(
    root=dataset_root,
    subset=1.0,
    url="https://cernbox.cern.ch/s/PYurUUzcNdXEGpz/download",
    pre_filter=graph_filter,
    pre_transform=None,
    post_filter=None,
    verbose=True
)

trainset, testset = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False)

# ---- MODEL ----
model = Transformer(
    input_size, hidden_size, encoding_size, g_norm, heads,
    num_xprtz, xprt_size, k, dropout_encoder, layers, output_size
).to(device)

model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
model.eval()

criterion = torch.nn.CrossEntropyLoss()
loss = LoadBalancingLoss(criterion, 0)

# ---- EVALUATION ----
result = evaluate(test_loader, model, loss, device)
predictions = torch.argmax(torch.cat(result[0]), dim=1)
truths = torch.cat(result[1])
which_node = torch.cat(result[4], dim=-1).cpu()

# Save evaluation metrics
acc = accuracy_score(truths.cpu(), predictions.cpu())
precision = precision_score(truths.cpu(), predictions.cpu())
recall = recall_score(truths.cpu(), predictions.cpu())
test_auroc_sklearn = roc_auc_score(truths.cpu(), torch.cat(result[0])[:, 1].cpu())

confmat = confusion_matrix(truths.cpu(), predictions.cpu())
print(f"Accuracy: {acc:.4f}, AUROC: {test_auroc_sklearn:.4f}")

np.savez_compressed(os.path.join(output_dir, 'eval_outputs.npz'),
                    predictions=predictions.cpu().numpy(),
                    truths=truths.cpu().numpy(),
                    which_node=which_node.numpy(),
                    acc=acc, precision=precision, recall=recall,
                    confmat=confmat)

# ---- METADATA FOR VISUALIZATION ----
graph_labels = []
node_counts = []
node_type_ids = []

for i in testset:
    label = int(i.y.item())
    graph_labels.append(label)
    num_nodes = i.x.shape[0]
    node_counts.append(num_nodes)
    node_type_ids.extend(list(range(num_nodes)))  # Sequential node types

np.savez_compressed(os.path.join(output_dir, 'test_metadata.npz'),
                    graph_labels=np.array(graph_labels, dtype=np.int32),
                    node_counts=np.array(node_counts, dtype=np.int32),
                    node_type_ids=np.array(node_type_ids, dtype=np.int32))

print(f"Saved evaluation metadata to {output_dir}")
