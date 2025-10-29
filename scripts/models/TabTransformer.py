"""
NumericalTransformer

This module implements a Transformer-based classifier for purely numerical
tabular data. We project the scalar value to an embedding vector, apply a
stack of self-attention layers (TransformerEncoder), flatten the sequence,
and classify with an MLP.

Key details:
- A single Linear(1 -> emb_dim) is applied to each feature value to form
  its token embedding (shared weights across features).
- TransformerEncoder operates on shape [batch, num_features, emb_dim]
  (batch_first=True).
- The flattened representation is passed to an MLP that outputs logits.
- The module returns raw logits; `predict_proba` provides softmax probs.

Citation:
Huang, X., et al. TabTransformer: Tabular Data Modeling Using Contextual
Embeddings. arXiv:2012.06678. DOI: 10.48550/arXiv.2012.06678(2020).

10/24/2025 --- SD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes=2, emb_dim=32, num_heads=4, num_layers=2, mlp_hidden=[128, 64], dropout=0.1):
        super().__init__()

        # Applies to each feature value individually: [B, F, 1] -> [B, F, emb_dim]
        self.embedding = nn.Linear(1, emb_dim)

        # Transformer encoder over embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP classifier
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential()
        input_dim = num_features * emb_dim
        for h in mlp_hidden:
            self.mlp.append(nn.Linear(input_dim, h))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))
            input_dim = h
        self.mlp.append(nn.Linear(input_dim, num_classes))

    def forward(self, x_cont):
        """
        Forward pass of the NumericalTransformer model.

        :param x_cont: Continuous features of shape [B, F] (Tensor)
        :return: Output logits of shape [num_nodes, num_classes] (Tensor)
        """

        if not isinstance(x_cont, torch.Tensor):
            x_cont = torch.tensor(x_cont, dtype=torch.float32)

        # x_cont: [B, num_features]
        B, F = x_cont.shape
        x = x_cont.view(B, F, 1)  # [B, F, 1]
        x = self.embedding(x)    # [B, F, emb_dim]
        x = self.transformer(x)  # [B, F, emb_dim]
        x = self.flatten(x)      # [B, F * emb_dim]

        return self.mlp(x)

    def predict_proba(self, x_cont):
        """
        Returns class probabilities via softmax over logits.

        :param x_cont: Input features of shape [B, F] (Tensor)
        :return: Array of shape [B, num_classes] with probabilities that sum to 1 (Numpy Array)
        """

        self.eval()

        with torch.no_grad():
            logits = self.forward(x_cont)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()