import os
import numpy as np
import torch

from data.tensordataset import TensorDataset
from geomstats.geometry.hypersphere import Hypersphere

class CSVDataset(TensorDataset):
    def __init__(self, file, delimiter=",", skip_header=1):
        data = np.genfromtxt(file, delimiter=delimiter, skip_header=skip_header)
        super().__init__(data)


class SphericalDataset(CSVDataset):
    def __init__(self, file, extrinsic=False, delimiter=",", skip_header=1):
        super().__init__(file, delimiter=delimiter, skip_header=skip_header)

        self.manifold = Hypersphere(2)
        self.intrinsic_data = (
            np.pi * (self.data / 180.0) + np.array([np.pi / 2, np.pi])[None, :]
        )
        self.intrinsic_data = torch.from_numpy(self.intrinsic_data).float()
        self.data = self.manifold.spherical_to_extrinsic(self.intrinsic_data)


class VolcanicErruption(SphericalDataset):
    def __init__(self, data_dir="data/earth", **kwargs):
        super().__init__(os.path.join(data_dir, "volerup.csv"), skip_header=2)


class Fire(SphericalDataset):
    def __init__(self, data_dir="data/earth", **kwargs):
        super().__init__(os.path.join(data_dir, "fire.csv"))


class Flood(SphericalDataset):
    def __init__(self, data_dir="data/earth", **kwargs):
        super().__init__(os.path.join(data_dir, "flood.csv"), skip_header=2)


class Earthquake(SphericalDataset):
    def __init__(self, data_dir="data/earth", **kwargs):
        super().__init__(os.path.join(data_dir, "quakes_all.csv"), skip_header=4)