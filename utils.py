import torch.nn as nn
import torch

# Define an embedding and encoding module
class EmbedEncode(nn.Module):
    def __init__(self, input_size, hidden_size, encoding_size, g_norm):
        super().__init__()
        # Linear transformation for input embedding
        self.embedding = nn.Linear(input_size, hidden_size)
        # Linear transformation for positional encoding
        self.encoding = nn.Linear(encoding_size, hidden_size)
        # Boolean flag to determine which input data to use
        self.g_norm = g_norm

    def forward(self, x):
        # If g_norm is True, use normalized data; otherwise, use raw input
        if self.g_norm == True:
            x = self.embedding(x.data_norm) 
        else:
            x = self.embedding(x.x) 

        return x


# Define a custom loss function with load balancing
class LoadBalancingLoss(nn.Module):
    def __init__(self, criterion, w_load):
        super().__init__()
        # Base loss function (e.g., CrossEntropyLoss, MSELoss)
        self.crit = criterion
        # Weighting factor for the load balancing term
        self.w_load = w_load

    def forward(self, output, labels):
        # Unpack model outputs (predictions, load balancing values, and unused variables)
        predictions, loads, discard, discard, discard_ = output
        # Compute standard loss using the criterion
        C = self.crit(predictions, labels)
        # Compute load balancing loss component
        LBL = self.w_load * loads

        # Return the combined loss
        return C + LBL

def normalized_entropy(p, base=None):
    """
    Computes entropy normalized to [0, 1] by dividing by maximum entropy (log(K)).
    :param p: Tensor of shape [batch_size, num_classes], softmax outputs
    :param base: Log base (e.g. 2, e, 10). Default: natural log
    :return: Tensor of shape [batch_size] with normalized entropy values
    """
    eps = 1e-16
    K = p.size(1)

    if base is None:
        entropy = -(p * (p + eps).log()).sum(dim=1)
        max_entropy = torch.log(torch.tensor(K, dtype=torch.float32, device=p.device))
    else:
        base = torch.tensor(base, dtype=torch.float32, device=p.device)
        entropy = -(p * (p + eps).log() / base.log()).sum(dim=1)
        max_entropy = torch.log(torch.tensor(K, dtype=torch.float32, device=p.device)) / base.log()

    return entropy / max_entropy




def enable_dropout(model):
    """ Function to enable the dropout layers during inference """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            
def check_dropout_active(model):
    active = any(m.training and isinstance(m, torch.nn.Dropout) for m in model.modules())
    print(f"[INFO] Dropout enabled during eval: {active}")
    return active
