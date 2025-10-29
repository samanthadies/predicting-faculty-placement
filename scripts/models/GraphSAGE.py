"""
GraphSAGE

This module implements a GraphSAGE model for node-level prediction tasks.
GraphSAGE (Graph Sample and Aggregate) learns node embeddings by aggregating
information from neighbors, allowing for inductive learning on unseen nodes.

Key Details:
- Hidden layers use SAGEConv with ReLU activations.
- Dropout is applied after each hidden layer for regularization.
- The final output layer produces logits for each class without activation.

Citation:
Hamilton, W., Ying, Z. & Leskovec, J. Inductive representation
learning on large graphs. In Advances in Neural Information
Processing Systems, vol. 30, DOI: 10.5555/3294771.3294869
(Curran Associates, Inc., 2017).

10/24/2025 --- SD
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        in_channels = num_features

        # construct hidden layers
        for hidden_dim in hidden_channels:
            self.convs.append(SAGEConv(in_channels, hidden_dim))
            in_channels = hidden_dim

        # final layer
        self.convs.append(SAGEConv(in_channels, num_classes))

    def forward(self, x, edge_index):
        """
        Forward pass of the GraphSAGE model.

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