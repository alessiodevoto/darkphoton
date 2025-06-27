import sys
import os
import torch
import numpy as np
import random
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

from dataset.darkphotondataset import DarkPhotonDataset
from dataset.transforms import GraphFilter
from models.model import Transformer
from utils import LoadBalancingLoss, normalized_entropy, enable_dropout, check_dropout_active
from evaluation.metrics import eceloss, uceloss
from train.train import evaluate

# Import BinaryCalibrationError metric
from torchmetrics.classification import BinaryCalibrationError

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
output_dir = './evaluation_outputs_mc'
num_mc_passes = 50

os.makedirs(output_dir, exist_ok=True)

# ---- SET SEEDS ----
def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seeds(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- LOAD DATA ----
dataset_root = './data'
os.makedirs(dataset_root, exist_ok=True)

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
test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False)

# ---- MODEL ----
model = Transformer(
    input_size, hidden_size, encoding_size, g_norm, heads,
    num_xprtz, xprt_size, k, dropout_encoder, layers, output_size
).to(device)

model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
model.eval()
enable_dropout(model)
print([module for module in model.modules() if isinstance(module, torch.nn.Dropout)])
check_dropout_active(model)

criterion = torch.nn.CrossEntropyLoss()
loss = LoadBalancingLoss(criterion, 0)

# Initialize BinaryCalibrationError metrics for different norms
bce_l1 = BinaryCalibrationError(n_bins=15, norm='l1').to(device)
bce_l2 = BinaryCalibrationError(n_bins=15, norm='l2').to(device)
bce_max = BinaryCalibrationError(n_bins=15, norm='max').to(device)

# ---- MONTE CARLO DROPOUT EVALUATION ----
all_mc_softmaxes = []
all_labels = None

for mc_pass in range(num_mc_passes):
    print(f"Running MC pass {mc_pass + 1}/{num_mc_passes}")
    set_seeds(seed + mc_pass)
    enable_dropout(model)
    check_dropout_active(model)
    result = evaluate(test_loader, model, loss, device)

    logits = torch.cat(result[0])
    labels = torch.cat(result[1])
    predictions = torch.argmax(logits, dim=1)
    acc = accuracy_score(labels.cpu(), predictions.cpu())
    precision = precision_score(labels.cpu(), predictions.cpu())
    recall = recall_score(labels.cpu(), predictions.cpu())
    auroc = roc_auc_score(labels.cpu(), logits[:, 1].cpu())
    confmat = confusion_matrix(labels.cpu(), predictions.cpu())

    softmaxes = torch.nn.functional.softmax(logits, dim=1)

    if all_labels is None:
        all_labels = labels

    all_mc_softmaxes.append(softmaxes.unsqueeze(0))  # (1, N, C)

    np.savez_compressed(os.path.join(output_dir, f'eval_outputs_{seed}.npz'),
                        predictions=predictions.cpu().numpy(),
                        truths=labels.cpu().numpy(),
                        prediction_softmax=softmaxes.cpu().numpy(),
                        acc=acc, precision=precision, recall=recall,
                        confmat=confmat)
    print(f"[Seed {seed + mc_pass}] Accuracy: {acc:.4f}, AUROC: {auroc:.4f}")

# Stack all softmaxes: (num_mc_passes, N, C)
all_mc_softmaxes = torch.cat(all_mc_softmaxes, dim=0)

# Compute mean and std over MC passes
mean_softmax = all_mc_softmaxes.mean(dim=0)  # (N, C)
std_softmax = all_mc_softmaxes.std(dim=0)    # (N, C)

# ---- CALIBRATION METRICS ----
ece, acc_in_bin, avg_conf_in_bin = eceloss(mean_softmax, all_labels)
uce, err_in_bin, avg_entropy_in_bin = uceloss(mean_softmax, all_labels)

# Compute BinaryCalibrationError for different norms using positive class probability
positive_probs = mean_softmax[:, 1]
bce_val_l1 = bce_l1(positive_probs, all_labels)
bce_val_l2 = bce_l2(positive_probs, all_labels)
bce_val_max = bce_max(positive_probs, all_labels)

print(f"\n=== MC-Dropout Calibration Metrics ===")
print(f"ECE: {ece.item():.4f}")
print(f"UCE: {uce.item():.4f}")
print(f"BCE (l1 norm): {bce_val_l1.item():.4f}")
print(f"BCE (l2 norm): {bce_val_l2.item():.4f}")
print(f"BCE (max norm): {bce_val_max.item():.4f}")

np.savez_compressed(os.path.join(output_dir, f'mcdropout_calibration.npz'),
                    ece=ece.cpu().numpy(),
                    uce=uce.cpu().numpy(),
                    bce_l1=bce_val_l1.cpu().numpy(),
                    bce_l2=bce_val_l2.cpu().numpy(),
                    bce_max=bce_val_max.cpu().numpy(),
                    acc_in_bin=acc_in_bin.cpu().numpy(),
                    avg_conf_in_bin=avg_conf_in_bin.cpu().numpy(),
                    err_in_bin=err_in_bin.cpu().numpy(),
                    avg_entropy_in_bin=avg_entropy_in_bin.cpu().numpy())

# ---- METADATA ----
graph_labels = []
node_counts = []
node_type_ids = []

for i in testset:
    graph_labels.append(int(i.y.item()))
    node_counts.append(i.x.shape[0])
    node_type_ids.extend(range(i.x.shape[0]))

np.savez_compressed(os.path.join(output_dir, f'mcdropout_test_metadata.npz'),
                    graph_labels=np.array(graph_labels, dtype=np.int32),
                    node_counts=np.array(node_counts, dtype=np.int32),
                    node_type_ids=np.array(node_type_ids, dtype=np.int32))

print(f"Saved all MC Dropout evaluation outputs to: {output_dir}")
