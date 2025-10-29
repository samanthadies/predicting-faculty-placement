"""
GCN

This module implements a Graph Convolutional Network (GCN) for node-level prediction tasks.
The architecture is flexible, allowing the user to specify:
- Number and size of hidden layers
- Dropout rate for regularization

Key Details:
- Hidden layers use GCNConv layers with ReLU activations.
- Dropout is applied after each hidden layer for regularization.
- The final output layer produces logits for each class without an activation.

Citation:
Kipf, T. N. & Welling, M. Semi-supervised classification with
graph convolutional networks. arXiv preprint arXiv:1609.02907
DOI: 10.48550/arXiv.1609.02907 (2016).

10/24/2025 --- SD
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        in_channels = num_features

        # construct hidden layers
        for hidden_dim in hidden_channels:
            self.convs.append(GCNConv(in_channels, hidden_dim))
            in_channels = hidden_dim

        # final layer
        self.convs.append(GCNConv(in_channels, num_classes))

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        :param x: Node feature matrix of shape [num_nodes, num_features] (Tensor)
        :param edge_index: Graph connectivity (COO format) of shape [2, num_edges] (LongTensor)
        :return: Output logits of shape [num_nodes, num_classes] (Tensor)
        """

        # iterate over hidden layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # apply the final layer without activation or dropout (logits)
        x = self.convs[-1](x, edge_index)

        return x