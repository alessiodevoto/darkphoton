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
from utils import LoadBalancingLoss, normalized_entropy, check_dropout_active
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
output_dir = './evaluation_outputs_noisy_routing'
seeds_to_test = [seed + i for i in range(50)]

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
check_dropout_active(model)

criterion = torch.nn.CrossEntropyLoss()
loss = LoadBalancingLoss(criterion, 0)

# Initialize BinaryCalibrationError metrics for different norms
bce_l1 = BinaryCalibrationError(n_bins=15, norm='l1').to(device)
bce_l2 = BinaryCalibrationError(n_bins=15, norm='l2').to(device)
bce_max = BinaryCalibrationError(n_bins=15, norm='max').to(device)

# ---- EVALUATION ----
all_softmaxes = []
all_labels = []

for seed in seeds_to_test:
    set_seeds(seed)
    print('seed', seed)
    result = evaluate(test_loader, model, loss, device, use_noisy_routing=True)

    logits = torch.cat(result[0])
    labels = torch.cat(result[1])
    softmaxes = torch.nn.functional.softmax(logits, dim=1)

    all_softmaxes.append(softmaxes)
    all_labels.append(labels)

    predictions = torch.argmax(logits, dim=1)
    acc = accuracy_score(labels.cpu(), predictions.cpu())
    precision = precision_score(labels.cpu(), predictions.cpu())
    recall = recall_score(labels.cpu(), predictions.cpu())
    auroc = roc_auc_score(labels.cpu(), logits[:, 1].cpu())
    confmat = confusion_matrix(labels.cpu(), predictions.cpu())

    np.savez_compressed(os.path.join(output_dir, f'eval_outputs_{seed}.npz'),
                        predictions=predictions.cpu().numpy(),
                        truths=labels.cpu().numpy(),
                        prediction_softmax=softmaxes.cpu().numpy(),
                        acc=acc, precision=precision, recall=recall,
                        confmat=confmat)
    print(f"[Seed {seed}] Accuracy: {acc:.4f}, AUROC: {auroc:.4f}")

# ---- CALIBRATION METRICS ----
softmaxes_all = torch.cat(all_softmaxes, dim=0)
labels_all = torch.cat(all_labels, dim=0)

ece, acc_in_bin, avg_conf_in_bin = eceloss(softmaxes_all, labels_all)
uce, err_in_bin, avg_entropy_in_bin = uceloss(softmaxes_all, labels_all)

# Compute BinaryCalibrationError for different norms using positive class probabilities
positive_probs = softmaxes_all[:, 1]
bce_val_l1 = bce_l1(positive_probs, labels_all)
bce_val_l2 = bce_l2(positive_probs, labels_all)
bce_val_max = bce_max(positive_probs, labels_all)

print(f"\n=== Aggregated Calibration Metrics ===")
print(f"ECE: {ece.item():.4f}")
print(f"UCE: {uce.item():.4f}")
print(f"BCE (l1 norm): {bce_val_l1.item():.4f}")
print(f"BCE (l2 norm): {bce_val_l2.item():.4f}")
print(f"BCE (max norm): {bce_val_max.item():.4f}")

np.savez_compressed(os.path.join(output_dir, f'calibration_aggregated.npz'),
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

np.savez_compressed(os.path.join(output_dir, f'test_metadata.npz'),
                    graph_labels=np.array(graph_labels, dtype=np.int32),
                    node_counts=np.array(node_counts, dtype=np.int32),
                    node_type_ids=np.array(node_type_ids, dtype=np.int32))

print(f"Saved all outputs and calibration metrics to: {output_dir}")



