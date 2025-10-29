"""
GConvGRU

This module implements a Recurrent Graph Convolutional Network using GConvGRU
cells from PyTorch Geometric Temporal for spatiotemporal graph modeling.

Key Details:
- Combines graph convolution with gated recurrent units (GRU) for temporal modeling.
- Hidden "spatial" GConvGRU layers process node features without temporal memory.
- A final "temporal" GConvGRU layer can use a hidden state (H) to capture temporal dynamics.
- Batch normalization and ReLU activation are applied after each GConvGRU layer.
- Dropout is applied after hidden layers for regularization.
- Output layer is a linear projection producing logits or continuous outputs.

Citation:
Seo, Y., Defferrard, M., Vandergheynst, P. & Bresson, X. Structured
sequence modeling with graph convolutional recurrent networks. In
Neural Information Processing: 25th International Conference, 362–373,
DOI: 10.1007/978-3-030-04167-0_33 (Springer, 2018).

10/24/2025 --- SD
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU


class RecurrentGCN(nn.Module):
    def __init__(self, in_channels, out_channels, K, hidden_channels=[64], dropout=0.5):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # spatial GConvGRU layers (no temporal memory) + BatchNorm
        self.spatial_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()

        input_dim = in_channels
        for hidden_dim in hidden_channels:
            self.spatial_layers.append(GConvGRU(input_dim, hidden_dim, K=K))
            self.batchnorm_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

        # temporal GConvGRU layer (can use memory state H)
        self.temporal_layer = GConvGRU(input_dim, out_channels, K=K)
        self.temporal_bn = nn.BatchNorm1d(out_channels)

        # final linear
        self.output_layer = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, H=None):
        """
        Forward pass of the RecurrentGCN model.

        :param x: Node feature matrix of shape [num_nodes, in_channels] (Tensor)
        :param edge_index: Graph connectivity (COO format) of shape [2, num_edges] (LongTensor)
        :param edge_weight: Edge weights of shape [num_edges] or None (Tensor)
        :param H:  Hidden state for temporal GConvGRU of shape [num_nodes, out_channels] (Tensor, optional)
        :return: Tuple[Tensor, Tensor]:
                - out (Tensor): Output logits of shape [num_nodes, out_channels]
                - h (Tensor): Updated hidden state of shape [num_nodes, out_channels]
        """

        h = x

        # spatial GConvGRU layers (no temporal memory)
        for i, layer in enumerate(self.spatial_layers):
            h = layer(h, edge_index, edge_weight, H=None)  # No memory
            h = self.batchnorm_layers[i](h)
            h = F.relu(h)
            h = self.dropout(h)

        # temporal GConvGRU layer (uses optional memory H)
        h = self.temporal_layer(h, edge_index, edge_weight, H=H)
        h = self.temporal_bn(h)
        h = F.relu(h)

        # linear projection to output logits
        out = self.output_layer(h)

        return out, h
