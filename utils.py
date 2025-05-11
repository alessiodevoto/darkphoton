import torch.nn as nn

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
            x = self.embedding(x.data_norm) + self.encoding(x.pe)
        else:
            x = self.embedding(x.x) + self.encoding(x.pe)

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
        predictions, loads, discard, discard_ = output
        # Compute standard loss using the criterion
        C = self.crit(predictions, labels)
        # Compute load balancing loss component
        LBL = self.w_load * loads

        # Return the combined loss
        return C + LBL

