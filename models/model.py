from torch import nn
import torch
from utils import EmbedEncode
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj
import numpy as np

"""
The MHGAttend class implements a multi-head graph attention mechanism, a specialized neural network layer designed for graph-structured data. 
This class takes as input a feature matrix of nodes and an adjacency matrix that defines the graph structure, and computes attention-weighted node embeddings.

Inputs:
- x (Tensor): Node feature matrix of shape (node_count, hidden_size).
- Adj (Tensor): Adjacency matrix of shape (node_count, node_count), with 0 indicating no edge and 1 indicating an edge.

Outputs:
- output (Tensor): Updated node embeddings of shape (node_count, hidden_size).
- attention_scores (Tensor): Attention weights of shape (heads, node_count, node_count).

Parameters:
- hidden_size (int): Dimensionality of the node embeddings.
- heads (int): Number of attention heads.
"""

class MHGAttend(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.hidden_size = hidden_size 
        self.heads = heads
        self.head_size = hidden_size//heads
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, Adj):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.view(q.shape[0], self.heads, self.head_size).transpose(0,1)  # \
        k = k.view(k.shape[0], self.heads, self.head_size).transpose(0,1)  #  |=>  (heads, nodebatch, head_size) compute the linear projection
        v = v.view(v.shape[0], self.heads, self.head_size).transpose(0,1)  # /
        attention_scores = (q @ v.transpose(-2,-1)) / np.sqrt(self.head_size)  #  (heads,nodebatch,head_size)*(heads,head_size,nodebatch) 
        attention_scores.masked_fill_(Adj == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        output = attention_scores @ v  #  (heads,nodebatch,nodebatch)*(heads,nodebatch,head_size)
        output = output.transpose(0,1).contiguous().view(output.shape[1], self.hidden_size)  #  (nodebatch,hidden_size)

        return self.wo(output), attention_scores


"""
The MoeveForward class implements a Mixture-of-Experts (MoE) model with a forward pass that utilizes a stochastic routing mechanism. 
This architecture is designed to dynamically route inputs to a subset of specialized experts.

Key Features:
- Multiple Experts: Each expert is a small feedforward neural network (FFN) that processes input data independently.
- Stochastic Routing: Inputs are routed to the top-k experts based on routing logits, which include added noise for stochasticity.
- Load Balancing: Tracks the load and sample assignments for each expert to monitor utilization.
- Noise Network: Introduces randomness in the routing process for exploration and diversity.

Inputs:
- input (Tensor): Input tensor of shape (batch_size, input_size).

Outputs:
- output_tensor (Tensor): Aggregated output tensor of shape (batch_size, input_size), combining contributions from the selected experts.
- expert_loads (Tensor): Load distribution tensor of shape (num_experts), representing the computational load for each expert.
- expert_sample_count (Tensor): Tensor of shape (num_experts) tracking the number of samples processed by each expert.
- sample_to_expert_assignment (Tensor): Tensor of shape (num_experts, batch_size) indicating the assignment of samples to experts.

Parameters:
- num_experts (int): Number of experts in the mixture.
- input_size (int): Dimensionality of the input data.
- xprt_size (int): Dimensionality of the hidden layer within each expert.
- k (int): Number of top-k experts selected for processing each input.
- dropout (float): Dropout rate for regularization within each expert.
"""
class MoeveForward(nn.Module):
    def __init__(self, num_experts, input_size, xprt_size, k, dropout):
        super().__init__()
        
        # Define a single expert: A simple feedforward neural network (FFN)
        self.expert = nn.Sequential(
            nn.Linear(input_size, xprt_size),  # Linear layer: input -> hidden
            nn.LeakyReLU(),                    # Activation function
            nn.Linear(xprt_size, input_size),  # Linear layer: hidden -> output
            nn.Dropout(dropout)                # Dropout for regularization
        )
        
        
        # Combine multiple experts using ModuleList
        self.experts = nn.ModuleList([self.expert for _ in range(num_experts)])

        # Define a noise network to introduce randomness in the routing process
        self.noise_network = nn.Linear(input_size, num_experts, bias=False)

        # Define the router network that decides which experts to activate
        self.router_network = nn.Linear(input_size, num_experts, bias=False)

        # Activation function to ensure noise values are non-negative
        self.softplus = nn.Softplus()

        # Save the number of experts and the top-k selection parameter
        self.num_experts = num_experts  # Number of experts
        self.k = k  # Number of top-k experts to select

        # Initialize the weights of the noise network to zero
        torch.nn.init.zeros_(self.noise_network.weight)

    def forward(self, input):
        
        device = input.device
        # Compute the routing logits from the router network
        router_logits = self.router_network(input)

        # Compute the noise values using the noise network and Softplus activation
        router_noise = self.softplus(self.noise_network(input))

        # Add random noise to the router's logits to introduce stochasticity
        routing_logits_noisy = router_logits + torch.randn(self.num_experts, device=device) * router_noise #(input_size,num_experts)

        # Select the top-k experts based on the noisy routing logits
        topk_weights, selected_experts = torch.topk(routing_logits_noisy, self.k) # `weights`: top-k values, `experts`: top-k indices #(input_size, k)

        # Apply a softmax to the selected weights to normalize them
        topk_weights = nn.functional.softmax(topk_weights, dim=1, dtype=torch.float)

        # Initialize outputs and tracking tensors
        output_tensor = torch.zeros_like(input, device=device) # Output tensor (same shape as input)
        expert_loads = torch.zeros(self.num_experts, device=device)  # Tracks load for each expert
        expert_sample_count = torch.zeros(self.num_experts, device=device)  # Tracks number of samples processed by each expert
        sample_to_expert_assignment = torch.zeros(self.num_experts, input.shape[0], device=device) # Tracks which samples go to which expert

        # Loop through each expert to compute outputs and loads
        for expert_idx, expert_network in enumerate(self.experts):
            # Identify samples assigned to the current expert
            assigned_nodes, selected_idx = torch.where(selected_experts == expert_idx)

            # Compute the output for the assigned samples
            output_tensor[assigned_nodes] += topk_weights[assigned_nodes, selected_idx, None] * expert_network(input[assigned_nodes])

            # Track which samples were assigned to this expert
            sample_to_expert_assignment[expert_idx][assigned_nodes] += 1

            # Calculate the load for the current expert
            # Find the k-th largest routing logit excluding the current expert
            kth_best_logits = torch.topk(
                torch.cat([
                    routing_logits_noisy.transpose(0, 1)[:expert_idx],         # Logits before the current expert
                    routing_logits_noisy.transpose(0, 1)[expert_idx+1:]     # Logits after the current expert
                ]).transpose(0, 1), self.k  # Select k-th largest logit
            )[0].transpose(0, 1)[self.k-1]
            # Retrieve the routing logit for the current expert
            router_logits_expert = router_logits.transpose(0, 1)[expert_idx]


            # Define a normal distribution (mean=0, std=1) for computing probabilities
            normal = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))

            # Compute the probability of the current expert being selected
            expert_selection_prob = normal.cdf((router_logits_expert - kth_best_logits) / router_noise.transpose(0, 1)[expert_idx])

            # Compute the load for the current expert
            load_i = torch.sum(expert_selection_prob)
            expert_loads[expert_idx] += load_i

            # Track the number of samples processed by this expert
            expert_sample_count[expert_idx] += len(assigned_nodes)

        # Return the output, loads, sample counts, and assignment tracking
        return output_tensor, expert_loads, expert_sample_count, sample_to_expert_assignment



#this class define the encoder network 
"""
The Encoder class implements a modular architecture for processing graph-structured data using attention mechanisms and Mixture-of-Experts (MoE) routing. 
Key Features:
- Multi-Head Graph Attention (MHGAttend): Captures dependencies between nodes in a graph by leveraging attention mechanisms.
- Mixture-of-Experts (MoeveForward): Dynamically routes inputs to a subset of specialized experts, enabling computational efficiency and diverse representations.

Inputs:
- x (Tensor): Input tensor of shape (num_nodes, hidden_size), where `num_nodes` is the number of nodes in the graph.
- Adj (Tensor): Adjacency matrix of shape (num_nodes, num_nodes), representing graph structure.

Outputs:
- y (Tensor): Output tensor of shape (num_nodes, hidden_size), containing the processed node representations.
- load_balance (Tensor): the squared normalized variance of expert_loads.
- expert_sample_count (Tensor): Tensor of shape (num_xprtz), tracking the number of samples processed by each expert.
- sample_to_expert_assignment (Tensor): Tensor of shape (num_xprtz, num_nodes), indicating the assignment of nodes to experts.

Parameters:
- hidden_size (int): Dimensionality of the input and output node features.
- heads (int): Number of attention heads in the graph attention mechanism.
- num_experts (int): Number of experts in the Mixture-of-Experts module.
- expert_hidden_size (int): Dimensionality of the hidden layer within each expert.
- k (int): Number of top-k experts selected for processing each node.
- dropout (float): Dropout rate for regularization.
"""

class Encoder(nn.Module):
    def __init__(self, hidden_size, heads, num_experts, expert_hidden_size, k, dropout):
        super().__init__()
        self.attend = MHGAttend(hidden_size, heads)
        self.moveforward = MoeveForward(num_experts, hidden_size, expert_hidden_size, k, dropout)
        self.normalize1 = nn.LayerNorm(hidden_size)
        self.normalize2 = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, Adj):
        input = x
        x, _ = self.attend(x,Adj)
        x = self.normalize1(input + x)
        x = self.drop(x)
        y, expert_loads, expert_sample_count, sample_to_expert_assignment = self.moveforward(x)
        y = self.normalize2(x + y) 
        load_balance = (torch.std(expert_loads) / torch.mean(expert_loads))**2 #squared normalized variance of load
    
        return y, load_balance, expert_sample_count, sample_to_expert_assignment



"""
    The Transformer class defines the full architecture of a transformer model for processing graph-structured data. 

    Key Features:
    - Embedding: Converts raw input features into a higher-dimensional space using `EmbedEncode`.
    - Multi-layer Encoders: Processes the data through multiple `Encoder` layers to extract hierarchical features.
    - Pooling: Aggregates node-level information into a graph-level representation using global mean pooling.
    - Prediction: Outputs final predictions for graph-level tasks.

    Inputs:
    - x (Data): Graph data object containing node features, edge indices, and batch information.

    Outputs:
    - x (Tensor): Final output predictions of shape (batch_size, output_size).
    - average_load_balance (Tensor): Scalar tensor representing average load imbalance across all layers.
    - average_expert_std (Tensor): Average standard deviation of the number of samples processed by each expert across all layers.
    - all_sample_assignments (Tensor): Tensor tracking the assignment of nodes to experts across all layers.

    Parameters:
    - input_size (int): Dimensionality of the input node features.
    - hidden_size (int): Dimensionality of the hidden layers and node features.
    - encoding_size (int): Size of the encoded feature representation.
    - g_norm (float): Graph normalization factor.
    - heads (int): Number of attention heads in the graph attention mechanism.
    - num_xprtz (int): Number of experts in the Mixture-of-Experts module.
    - xprt_size (int): Dimensionality of the hidden layer within each expert.
    - k (int): Number of top-k experts selected for processing each node.
    - dropout_encoder (float): Dropout rate applied within the encoder layers.
    - layers (int): Number of encoder layers in the transformer.
    - output_size (int): Dimensionality of the final prediction output.
"""

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, encoding_size, g_norm, heads, num_xprtz, xprt_size, k, dropout_encoder, layers, output_size):
        super().__init__()
        self.embed = EmbedEncode(input_size, hidden_size, encoding_size, g_norm)  # Embedding layer
        self.encoders = nn.ModuleList([Encoder(hidden_size, heads, num_xprtz, xprt_size, k, dropout_encoder) for _ in range(layers)])  # Stack of encoders
        self.cut = nn.Linear(hidden_size, hidden_size // 2)  # Linear layer for dimensionality reduction
        self.relu = nn.ReLU()  # Activation function
        self.pool = global_mean_pool  # Global mean pooling for graph-level representation
        self.predict = nn.Linear(hidden_size // 2, output_size)  # Final prediction layer
        self.num_xprtz = num_xprtz  # Number of experts
        self.layers = layers  # Number of encoder layers

    def forward(self, x):
        batch_indices = x.batch  # Extract batch indices for graph-level pooling
        Adj = to_dense_adj(x.edge_index)  # Convert edge indices to dense adjacency matrix
        x = self.embed(x)  # Apply embedding layer

        # Initialize tensors on the same device as the input tensor
        device = x.device  # Get the device of the input tensor
        load_balances = torch.zeros(self.layers, device=device)  # Initialize load balancing tensor
        expert_std_devs = torch.zeros(self.layers, device=device)  # Initialize expert standard deviation tensor
        all_sample_assignments = torch.empty(0, x.shape[0], device=device)  # Initialize sample assignment tensor

        for i, encoder in enumerate(self.encoders):
            # Pass through each encoder layer
            x, load_balance, expert_sample_count, sample_to_expert_assignment = encoder(x, Adj)
            load_balances[i] += load_balance  # Accumulate load balance
            expert_std_devs[i] += torch.std(expert_sample_count)  # Accumulate expert standard deviation
            all_sample_assignments = torch.cat([all_sample_assignments, sample_to_expert_assignment], dim=0)  # Track sample assignments

        # Compute averages
        average_expert_std = torch.mean(expert_std_devs, dim=0)  # Compute average standard deviation
        average_load_balance = torch.mean(load_balances, dim=0)  # Compute average load balance

        # Classification head for prediction
        x = self.relu(self.cut(x))  # Apply dimensionality reduction and activation
        x = self.pool(x, batch_indices)  # Apply global mean pooling
        x = self.predict(x)  # Final prediction

        return x, average_load_balance, average_expert_std, all_sample_assignments
