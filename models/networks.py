import torch
import numpy as np
from models.layers import MLP


class ScoreNetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, hid_dim, out_dim, act, manifold, **kwargs):
        super().__init__()
        self.layer = MLP(num_layers, in_dim, hid_dim, out_dim, act)
        self.manifold = manifold

    def forward(self, x, t):
        if len(t.shape) == len(x.shape)-1:
            t = t.unsqueeze(-1)
        output = self.layer(torch.cat([x, t], dim=-1))
        return self.manifold.to_tangent(output, x)