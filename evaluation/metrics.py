import torch
from utils import normalized_entropy


def uceloss(softmaxes, labels, n_bins=15):
    '''softmaxes is the average of softmax over N passes
       softmaxes size: [batch_size, num_classes]'''
    d = softmaxes.device  # Get the device (CPU or GPU)

    # Create bin boundaries from 0 to 1, split into `n_bins` intervals
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    bin_lowers = bin_boundaries[:-1]  # Lower bounds of bins
    bin_uppers = bin_boundaries[1:]   # Upper bounds of bins

    # Get predicted class labels (index of max softmax value)
    _, predictions = torch.max(softmaxes, 1)
    # Boolean tensor indicating whether each prediction is wrong (1 = wrong, 0 = correct)
    errors = predictions.ne(labels)

    # Compute uncertainty using normalized entropy
    uncertainties = normalized_entropy(softmaxes, base=softmaxes.size(1))

    # Lists to store bin-wise stats
    errors_in_bin_list = []
    avg_entropy_in_bin_list = []

    uce = torch.zeros(1, device=d)  # Final uncertainty calibration error

    # Loop over each bin
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Select examples with uncertainty in the current bin
        in_bin = uncertainties.gt(bin_lower.item()) * uncertainties.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # Proportion of total samples in bin (|Bm| / n)

        if prop_in_bin.item() > 0.0:
            # Mean error rate in bin
            errors_in_bin = errors[in_bin].float().mean()
            # Mean uncertainty (entropy) in bin
            avg_entropy_in_bin = uncertainties[in_bin].mean()
            # Contribution to UCE: |uncertainty - error rate| × proportion in bin
            uce += torch.abs(avg_entropy_in_bin - errors_in_bin) * prop_in_bin

            # Save stats for analysis
            errors_in_bin_list.append(errors_in_bin)
            avg_entropy_in_bin_list.append(avg_entropy_in_bin)

    # Convert lists to tensors
    err_in_bin = torch.tensor(errors_in_bin_list, device=d)
    avg_entropy_in_bin = torch.tensor(avg_entropy_in_bin_list, device=d)

    return uce, err_in_bin, avg_entropy_in_bin


def eceloss(softmaxes, labels, n_bins=15):
    """
    Modified from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    '''softmaxes is the average of softmax over N passes
       softmaxes size: [batch_size, num_classes]'''
    d = softmaxes.device  # Use same device as input

    # Create bins between 0 and 1
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # For each prediction, get the highest predicted probability and its class index
    confidences, predictions = torch.max(softmaxes, 1)
    # Check if predictions are correct (1 = correct, 0 = wrong)
    accuracies = predictions.eq(labels)

    # Lists to store bin-wise stats
    accuracy_in_bin_list = []
    avg_confidence_in_bin_list = []

    ece = torch.zeros(1, device=d)  # Final expected calibration error

    # Loop over each confidence bin
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Mask: which predictions fall into the current bin based on confidence
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n

        if prop_in_bin.item() > 0.0:
            # Accuracy within the bin
            accuracy_in_bin = accuracies[in_bin].float().mean()
            # Mean confidence in the bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            # Contribution to ECE: |confidence - accuracy| × proportion
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            # Save stats
            accuracy_in_bin_list.append(accuracy_in_bin)
            avg_confidence_in_bin_list.append(avg_confidence_in_bin)

    # Convert lists to tensors
    acc_in_bin = torch.tensor(accuracy_in_bin_list, device=d)
    avg_conf_in_bin = torch.tensor(avg_confidence_in_bin_list, device=d)

    return ece, acc_in_bin, avg_conf_in_bin

def mutual_information(mc_softmaxes):
    '''mc_softmaxes: Tensor [N, batch_size, num_classes]
       labels:       Tensor [batch_size]'''
    eps = 1e-16
    mean_softmax = mc_softmaxes.mean(dim=0)  # [batch_size, num_classes]
    ent_mean = -torch.sum(mean_softmax * torch.log(mean_softmax.clamp(min=eps)), dim=1)
    ent_all = -torch.sum(mc_softmaxes * torch.log(mc_softmaxes.clamp(min=eps)), dim=2)
    mean_ent = ent_all.mean(dim=0)
    return ent_mean - mean_ent


def predictive_entropy(mc_softmaxes):
    """
    Compute predictive entropy of the averaged predictive distribution.

    Args:
        mc_softmaxes: Tensor of shape [N, batch_size, num_classes]
            Softmax probabilities from N stochastic forward passes.

    Returns:
        Tensor of shape [batch_size] containing predictive entropy per sample.
    """
    eps = 1e-16
    mean_softmax = mc_softmaxes.mean(dim=0)  # [batch_size, num_classes]
    pred_entropy = -torch.sum(mean_softmax * torch.log(mean_softmax.clamp(min=eps)), dim=1)  # [batch_size]
    return pred_entropy
