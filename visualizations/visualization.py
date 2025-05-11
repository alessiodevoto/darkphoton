import os
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch


def plot_grouped_bars_sub(xprtlsignal, dictiorevsignal, num_experts, colors, wid, ax, label):
    x = np.arange(len(dictiorevsignal))
    width = wid / num_experts
    offset = (num_experts - 1) * width / 2
    for i in range(num_experts):
        ax.bar(
            x - offset + i * width,
            xprtlsignal[i + 1],
            width,
            label=f"Expert {i + 1}",
            color=colors[i],
            edgecolor='black',
            linewidth=2  # Add border with thickness p2=2
        )
    ax.set(xticks=x, xticklabels=list(dictiorevsignal.values()))
    plt.xticks(fontsize=12)
    ax.set(yticks=[0, 1])
    plt.yticks(fontsize=12)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("Nodes processed", fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.plot()

def plot_grouped_bars(xprtlsignal, dictiorevsignal, num_experts, colors, sizex, sizey, wid, label):
    plt.figure(figsize=(sizex, sizey))
    x = np.arange(len(dictiorevsignal))
    width = wid / num_experts
    offset = (num_experts - 1) * width / 2

    for i in range(num_experts):
        plt.bar(
            x - offset + i * width,
            xprtlsignal[i + 1],
            width,
            label=f"Expert {i + 1}",
            color=colors[i],
            edgecolor='black',
            linewidth=2  # Add border with thickness p2=2
        )

    plt.xticks(x, list(dictiorevsignal.values()))
    plt.xticks(fontsize=12)
    plt.yticks([0, 1])
    plt.yticks(fontsize=12)
    plt.xlabel(label, fontsize=12)
    plt.ylabel("Nodes processed", fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_grouped_bars_save(xprtlsignal, dictiorevsignal, num_experts, colors, layers, wid, path):
    # Create the subplots
    fig, axs = plt.subplots(ncols=layers, nrows=1, figsize=(layers * 7, 7))  # Adjust width based on layers

    fig.suptitle("Expert specialization", fontsize=16)  # Main title
    x = np.arange(len(dictiorevsignal))
    width = wid / num_experts
    offset = (num_experts - 1) * width / 2

    # Loop through each layer (subplot)
    for i in range(layers):
        ax = axs[i]
        for j in range(num_experts):
            ax.bar(
                x - offset + j * width,
                xprtlsignal[j + 1 + i * num_experts],  # Adjust for layer-specific data
                width,
                label=f"Expert {j + 1}",
                color=colors[j],
                edgecolor="black",
                linewidth=2  # Add border with thickness p2=2
            )
        ax.set(xticks=x, xticklabels=list(dictiorevsignal.values()))
        ax.tick_params(axis='x', rotation=45)  # Rotate labels for better readability
        ax.set_yticks([0, 1])
        ax.set_xlabel("Node type", fontsize=12)
        ax.set_ylabel("Nodes processed", fontsize=12)
        ax.set_title(f"Layer {i + 1}", fontsize=14)  # Subtitle for each layer

    # Create custom legend for the heatmap values 1 and 0
    patches = [
        Patch(color=colors[0], label='Expert activated (1)', edgecolor='black'),
        Patch(color='white', label='Expert not activated (0)', edgecolor='black')  # Assuming 0 is white or background color
    ]

    # Position the legend outside the plot
    fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.05, 1), title="Legend", fontsize=12)

    # Adjust the layout to allow space for the legend
    plt.subplots_adjust(wspace=0.4, bottom=0.25, right=0.85)  # Adjust right to make space for the legend

    # Save the figure
    here = os.getcwd()
    fig.savefig(path)
    plt.show()
