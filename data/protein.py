import torch
import numpy as np
import pandas as pd

from data.tensordataset import TensorDataset
from geomstats.geometry.torus import Torus


class Top500(TensorDataset):
    def __init__(self, data_dir="data/top500", amino="General"):
        self.manifold = Torus(2)
        data = pd.read_csv(
            f"{data_dir}/aggregated_angles.tsv",
            delimiter="\t",
            names=["source", "phi", "psi", "amino"],
        )

        amino_types = ["General", "Glycine", "Proline", "Pre-Pro"]
        assert amino in amino_types, f"amino type {amino} not implemented"

        data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")

        data = torch.tensor(data % 360 * np.pi / 180)
        #NOTE: coordintes instead of angles
        self.data = torch.stack([torch.cos(data[:,0]), torch.sin(data[:,0]), 
                        torch.cos(data[:,1]), torch.sin(data[:,1])], dim=1)
        self.amino = amino

        expand_factors = {'General': 1, 'Glycine': 10, 'Proline': 18, 'Pre-Pro': 20}
        self.expand_factor = expand_factors[amino]

        super().__init__(self.data)


class RNA(TensorDataset):
    def __init__(self, data_dir="data/rna"):
        self.manifold = Torus(7)
        data = pd.read_csv(
            f"{data_dir}/aggregated_angles.tsv",
            delimiter="\t",
            names=[
                "source",
                "base",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "chi",
            ],
        )

        data = data[
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        ].values.astype("float32")

        data = torch.tensor(data % 360 * np.pi / 180)
        #NOTE: coordintes instead of angles
        coords = []
        for i in range(data.shape[1]):
            coords.extend([torch.cos(data[:,i]), torch.sin(data[:,i])])
        self.data = torch.stack(coords, dim=1)
        
        self.expand_factor = 14

        super().__init__(self.data)
