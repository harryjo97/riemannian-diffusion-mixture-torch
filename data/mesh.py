import os
import igl
import numpy as np
import torch
import pickle

from data.tensordataset import TensorDataset
from geomstats.geometry.mesh import Mesh

class MeshTarget(TensorDataset):
    def __init__(self, data_dir, file_name, k, trunc=200, device='cuda:0'):
        NAME = file_name.split('.obj')[0]
        obj_file = os.path.join(data_dir, file_name)
        v, f = igl.read_triangle_mesh(obj_file)
        
        # downsample for stanford bunny
        if file_name == 'bunny.obj':
            # normalize mesh
            v = v / 250.

        vt = torch.from_numpy(v).float().to(device)
        ft = torch.from_numpy(f).int().to(device)
        self.manifold = Mesh(dim=3, v=vt, f=ft, trunc=trunc)
        
        with open(os.path.join(data_dir, f'{NAME}_{k}.pkl'), "rb") as ff:
            data, _ = pickle.load(ff)

        super().__init__(torch.from_numpy(data).float())


class SpotTheCow(MeshTarget):
    def __init__(self, data_dir="data/mesh", k=-1, trunc=200, device='cuda:0'):
        file_name = "spot.obj"
        super().__init__(data_dir, file_name, k, trunc, device)

class StanfordBunny(MeshTarget):
    def __init__(self, data_dir="data/mesh", k=-1, trunc=200, device='cuda:0'):
        file_name = "bunny.obj"
        super().__init__(data_dir, file_name, k, trunc, device)