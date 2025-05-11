# plot_specialization.py
import numpy as np
import torch
from matplotlib import pyplot as plt
from visualization import plot_grouped_bars, plot_grouped_bars_sub

# CONFIG
layers = 2
num_xprtz = 6
output_dir = './evaluation_outputs'

# REVERSE DICTIONARIES
dictiorevsignal = {-1: "signal", -2: "ttbar", -3: "singletop"}
dictiorevparticle = {0: "jet3", 1: "jet1", 2: "jet2", 3: "b1", 4: "b2", 5: "lepton", 6: "energy"}

colors = ["floralwhite", "darksalmon", "midnightblue", "mediumaquamarine", "goldenrod", "plum", "darkorange", "gray", "black", "green", "brown", "purple"]

# LOAD SAVED DATA
eval_data = np.load(f'{output_dir}/eval_outputs.npz')
meta_data = np.load(f'{output_dir}/test_metadata.npz')
which_node = torch.tensor(eval_data['which_node'])

event_ids = meta_data['event_ids']
node_counts = meta_data['node_counts']

# MAP events to type index
event_dict = {"signal": -1, "ttbar": -2, "singletop": -3}
event_labels = np.concatenate([[event_dict[e]] * c for e, c in zip(event_ids, node_counts)])

# Expert specialization on event types
xprtlsignal = {}
for i in range(num_xprtz * layers):
    temp = []
    for j in range(3):
        temp.append(len(np.where(event_labels[(torch.where(which_node[i] == 1)[0])] == -j - 1)[0]))
    xprtlsignal[i + 1] = temp

# Expert specialization on node types (assumes 6 or 7 node graphs)
idlist1 = [1, 2, 3, 4, 5, 6]
idlist2 = [1, 2, 0, 3, 4, 5, 6]
node_type_ids = []
for count in node_counts:
    node_type_ids.extend(idlist2 if count == 7 else idlist1)
node_type_ids = np.array(node_type_ids)

# Assign node types to experts
xprtlparticle = {}
for i in range(num_xprtz * layers):
    temp = []
    for j in range(7):
        temp.append(len(torch.where(which_node.T[torch.where(torch.tensor(node_type_ids) == j)[0]].T[i] == 1)[0]))
    xprtlparticle[i + 1] = temp

# ---- PLOTS ----
if layers == 1:
    num_experts = len(xprtlsignal)
    fig1 = plot_grouped_bars(xprtlsignal, dictiorevsignal, num_experts, colors, 7, 5, 0.6, "Event type")
    fig1.savefig(f"{output_dir}/specialization_event_type.png", bbox_inches='tight')

    fig2 = plot_grouped_bars(xprtlparticle, dictiorevparticle, num_experts, colors, 12, 6, 0.8, "Node type")
    fig2.savefig(f"{output_dir}/specialization_node_type.png", bbox_inches='tight')

else:
    # Event type
    fig, axs = plt.subplots(nrows=layers, figsize=(11, layers * 5))
    temp = 0
    for i in range(layers):
        data = {j + 1: xprtlsignal[temp + j + 1] for j in range(num_xprtz)}
        axs[i].title.set_text(f'Encoder number {i + 1}')
        plot_grouped_bars_sub(data, dictiorevsignal, num_xprtz, colors, 0.6, axs[i], "Event type")
        axs[i].grid()
        temp += num_xprtz
    fig.tight_layout()
    fig.savefig(f"{output_dir}/specialization_event_type_layers.png", bbox_inches='tight')

    # Node type
    fig, axs = plt.subplots(nrows=layers, figsize=(12, layers * 5))
    temp = 0
    for i in range(layers):
        data = {j + 1: xprtlparticle[temp + j + 1] for j in range(num_xprtz)}
        axs[i].title.set_text(f'Encoder number {i + 1}')
        plot_grouped_bars_sub(data, dictiorevparticle, num_xprtz, colors, 0.7, axs[i], "Node type")
        axs[i].grid()
        temp += num_xprtz
    fig.tight_layout()
    fig.savefig(f"{output_dir}/specialization_node_type_layers.png", bbox_inches='tight')

