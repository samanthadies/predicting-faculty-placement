"""
GAT

This module implements a Graph Attention Network (GAT) for node-level prediction tasks.
The architecture is flexible, allowing the user to specify:
- Number and size of hidden layers
- Number of attention heads per hidden and output layer
- Dropout rate for regularization

Key Details:
- Hidden layers use multi-head attention with concatenated outputs.
- The final layer uses a single head with averaged attention (no concatenation) to produce logits.
- Activation: ELU after each hidden GAT layer.
- Dropout: Applied after activation in hidden layers for regularization.

Citation:
Velickovic, P. et al. Graph attention networks. Stat
1050, 10–48550, DOI: 10.48550/arXiv.1710.10903 (2017).

10/24/2025 --- SD
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5, heads_hidden=8, heads_output=1):
        super().__init__()
        self.dropout = dropout
        self.heads_hidden = heads_hidden
        self.heads_output = heads_output

        self.convs = torch.nn.ModuleList()
        in_channels = num_features

        # hidden layers with multiple heads
        for hidden_dim in hidden_channels:
            self.convs.append(GATConv(in_channels, hidden_dim, heads=self.heads_hidden))
            in_channels = hidden_dim * self.heads_hidden  # output is concatenated across heads

        # final layer with 1 head
        self.convs.append(GATConv(in_channels, num_classes, heads=self.heads_output, concat=False))  # no concat at output

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.

        :param x: Node feature matrix of shape [num_nodes, num_features] (Tensor)
        :param edge_index: Graph connectivity (COO format) of shape [2, num_edges] (Tensor)
        :return: Output logits of shape [num_nodes, num_classes] (Tensor)
        """

        # iterate over hidden layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # apply the final layer without activation or dropout (logits)
        x = self.convs[-1](x, edge_index)
        return x