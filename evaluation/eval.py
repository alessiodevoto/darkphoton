# evaluate_model.py
import torch
import numpy as np
import random
import os
import re
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric's DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from dataset import CustomEventsDataset2, MakeHomogeneous
from model import Transformer
from utils import LoadBalancingLoss
from train import evaluate

# ---- CONFIGURATION ----
encoding_size = 4
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
batchsize = 500
seed = 42
model_ckpt_path = '/hdd3/dongen/Desktop/Susy/Mixture-of-Expert-Graph-Transformer/results/12804True262020.022150060450.10.001Trueprova/final_model.pt'
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
train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False)

# ---- MODEL ----
model = Transformer(input_size, hidden_size, encoding_size, g_norm, heads, num_xprtz, xprt_size, k, dropout_encoder, layers, output_size).to(device)
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
confmat = confusion_matrix(truths.cpu(), predictions.cpu())

np.savez_compressed(os.path.join(output_dir, 'eval_outputs.npz'),
                    predictions=predictions.cpu().numpy(),
                    truths=truths.cpu().numpy(),
                    which_node=which_node.numpy(),
                    acc=acc, precision=precision, recall=recall,
                    confmat=confmat)

# Save metadata needed for plots
event_ids = []
node_counts = []
for i in testset:
    clean_id = re.sub(r'\d+', '', i.event_id).replace("_", "")
    event_ids.append(clean_id)
    node_counts.append(i.x.shape[0])

np.savez_compressed(os.path.join(output_dir, 'test_metadata.npz'),
                    event_ids=np.array(event_ids),
                    node_counts=np.array(node_counts, dtype=np.int32))

print(f"Saved evaluation data to {output_dir}")
