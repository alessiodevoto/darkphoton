# plot_specialization.py
import numpy as np
import torch
from matplotlib import pyplot as plt
from visualization import plot_grouped_bars, plot_grouped_bars_sub

# CONFIG
layers = 2
num_xprtz = 6
output_dir = './evaluation_outputs'

# LOAD SAVED DATA
eval_data = np.load(f'{output_dir}/eval_outputs.npz')
meta_data = np.load(f'{output_dir}/test_metadata.npz')

which_node = torch.tensor(eval_data['which_node'])
graph_labels = meta_data['graph_labels']
node_counts = meta_data['node_counts']
node_type_ids = meta_data['node_type_ids']

# EVENT LABEL MAPPING PER NODE
event_labels = np.concatenate([[label] * count for label, count in zip(graph_labels, node_counts)])

# Expert specialization on event type (binary classification: 0 or 1)
xprtlsignal = {}
for i in range(num_xprtz * layers):
    temp = []
    for j in range(2):
        mask = (event_labels == j)
        active_nodes = torch.where(which_node[i] == 1)[0]
        temp.append(np.sum(event_labels[active_nodes.numpy()] == j))
    xprtlsignal[i + 1] = temp

# Expert specialization on node type
node_type_ids = torch.tensor(node_type_ids)
unique_node_types = np.unique(node_type_ids)
node_type_map = {i: f"Node {i}" for i in unique_node_types}

xprtlparticle = {}
for i in range(num_xprtz * layers):
    temp = []
    for ntype in unique_node_types:
        mask = node_type_ids == ntype
        temp.append(int(torch.sum(which_node[i][mask])))
    xprtlparticle[i + 1] = temp

colors = ["floralwhite", "darksalmon", "midnightblue", "mediumaquamarine",
          "goldenrod", "plum", "darkorange", "gray", "black", "green", "brown", "purple"]

# ---- PLOTS ----
if layers == 1:
    num_experts = len(xprtlsignal)
    fig1 = plot_grouped_bars(xprtlsignal, {0: "Class 0", 1: "Class 1"}, num_experts, colors, 7, 5, 0.6, "Event type")
    fig1.savefig(f"{output_dir}/specialization_event_type.png", bbox_inches='tight')

    fig2 = plot_grouped_bars(xprtlparticle, node_type_map, num_experts, colors, 12, 6, 0.8, "Node type")
    fig2.savefig(f"{output_dir}/specialization_node_type.png", bbox_inches='tight')

else:
    # Event type plot (layered)
    fig, axs = plt.subplots(nrows=layers, figsize=(11, layers * 5))
    temp = 0
    for i in range(layers):
        data = {j + 1: xprtlsignal[temp + j + 1] for j in range(num_xprtz)}
        axs[i].title.set_text(f'Encoder {i + 1}')
        plot_grouped_bars_sub(data, {0: "Class 0", 1: "Class 1"}, num_xprtz, colors, 0.6, axs[i], "Event type")
        axs[i].grid()
        temp += num_xprtz
    fig.tight_layout()
    fig.savefig(f"{output_dir}/specialization_event_type_layers.png", bbox_inches='tight')

    # Node type plot (layered)
    fig, axs = plt.subplots(nrows=layers, figsize=(12, layers * 5))
    temp = 0
    for i in range(layers):
        data = {j + 1: xprtlparticle[temp + j + 1] for j in range(num_xprtz)}
        axs[i].title.set_text(f'Encoder {i + 1}')
        plot_grouped_bars_sub(data, node_type_map, num_xprtz, colors, 0.7, axs[i], "Node type")
        axs[i].grid()
        temp += num_xprtz
    fig.tight_layout()
    fig.savefig(f"{output_dir}/specialization_node_type_layers.png", bbox_inches='tight')
