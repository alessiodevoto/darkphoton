import matplotlib.pyplot as plt
import numpy as np

def plot_reliability_diagram(acc_in_bin, avg_conf_in_bin):
    # Convert tensors to numpy for plotting
    acc = acc_in_bin.cpu().numpy()
    conf = avg_conf_in_bin.cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.plot(conf, acc, marker='o', label='Model')
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_uncertainty_calibration(err_in_bin, avg_entropy_in_bin):
    err = err_in_bin.cpu().numpy()
    entropy = avg_entropy_in_bin.cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.plot(entropy, err, marker='o', label='Model')
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Normalized Predictive Entropy')
    plt.ylabel('Empirical Error Rate')
    plt.title('Uncertainty Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_distribution(values, title='Distribution', xlabel='Value', bins=30):
    vals = values.cpu().numpy()
    plt.figure(figsize=(7,5))
    plt.hist(vals, bins=bins, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.show()
