import torch
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
import numpy as np
import math
from scipy.special import logsumexp

from geomstats.geometry.special_orthogonal import _SpecialOrthogonalMatrices, \
                                                    _SpecialOrthogonal3Vectors

class UniformDistribution:
    """Uniform density on compact manifold"""

    def __init__(self, manifold):
        self.manifold = manifold

    def sample(self, shape, device):
        return self.manifold.random_uniform(n_samples=shape[0], device=device)

    def log_prob(self, z):
        return -np.ones([z.shape[0]]) * self.manifold.log_volume


class Wrapped:
    """Wrapped normal density on compact manifold"""

    def __init__(self, scale, batch_dims, manifold, mean_type, **kwargs):
        self.batch_dims = batch_dims
        self.manifold = manifold
        self.device = None

        if mean_type == 'random':
            self.mean = manifold.random_uniform(n_samples=1)
            self.mean = self.mean.reshape(1, -1)
        elif mean_type == 'hyperbolic':
            self.mean = self.manifold.identity.unsqueeze(0)
        elif mean_type == 'mixture':
            self.mean = kwargs['mean']
        else:
            raise NotImplementedError(f'mean_type: {mean_type} not implemented.')

        self.scale = torch.ones((self.mean.shape)) * scale if isinstance(scale, float) \
                    else torch.tensor(scale)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample(self.batch_dims, self.device)

    def sample(self, n_samples, device='cpu'):
        if not isinstance(n_samples, int):
            n_samples = n_samples[0]
        mean = self.mean.to(device).repeat(n_samples, 1)
        scale = self.scale.to(device).repeat(n_samples, 1)

        tangent_vec = self.manifold.random_normal_tangent(
            base_point=mean, n_samples=n_samples)
        tangent_vec = scale * tangent_vec

        samples = self.manifold.exp(tangent_vec, mean)
        return samples

    # Used for SO3 and hyperbolic
    def log_prob(self, samples):
        # flatten samples
        bs = samples.shape[0]
        samples = samples.reshape(bs, -1)
        device = samples.device

        zero = torch.zeros((self.manifold.dim), device=device)

        # refactor axis contenation / removal
        scale = self.scale.flatten().to(device)
        if self.scale.shape[-1] == self.manifold.dim+1: # hyperboloid
            scale = scale[1:]
        if isinstance(self.manifold, _SpecialOrthogonalMatrices):
            assert torch.allclose(self.scale, self.scale[0])
            scale = torch.ones_like(zero) * self.scale[0]
        multivariatenormaldiag = Independent(Normal(zero, scale), 1)

        mean = self.mean.to(device).repeat(bs,1)
        tangent_vec = self.manifold.metric.log(samples, mean)

        #NOTE: Fix SO3 logp
        tangent_vec = self.manifold.metric.transpback0(mean, tangent_vec)

        norm_pdf = multivariatenormaldiag.log_prob(tangent_vec)
        logdetexp = self.manifold.logdetexp(mean, samples)
        log_prob = norm_pdf - logdetexp
        return log_prob.detach().cpu().numpy()


class WrappedMixture:
    """Wrapped normal mixture density on compact manifold"""

    def __init__(self, scale, batch_dims, manifold, mean_type, **kwargs):
        self.batch_dims = batch_dims
        self.manifold = manifold
        self.device = None

        if mean_type == 'so3':
            assert isinstance(manifold, _SpecialOrthogonalMatrices)
            means = []
            self.centers = [[0.0, 0.0, 0.0], [0.0, 0.0, np.pi], [np.pi, 0.0, np.pi]] 
            for v in self.centers:
                s = _SpecialOrthogonal3Vectors().matrix_from_tait_bryan_angles(np.array(v))
                means.append(s.float().flatten())
            self.mean = torch.stack(means)
        elif mean_type == 'poincare_disk':
            self.mean = torch.tensor([[-0.8, 0.0],[0.8, 0.0],[0.0, -0.8],[0.0, 0.8]])
        elif mean_type == 'hyperboloid4':
            mean = torch.tensor([[-0.4, 0.0],[0.4, 0.0],[0.0, -0.4],[0.0, 0.4]])
            self.mean = self.manifold._ball_to_extrinsic_coordinates(mean)
        elif mean_type == 'hyperboloid6':
            hex = [[0., 2.], [math.sqrt(3), 1.], [math.sqrt(3), -1.], [0., -2.], 
                    [-math.sqrt(3), -1.], [-math.sqrt(3), 1.]]
            mean = torch.tensor(hex) * 0.3
            self.mean = self.manifold._ball_to_extrinsic_coordinates(mean)
        elif mean_type == 'test':
            self.mean = kwargs['mean']
        else:
            raise NotImplementedError(f'mean_type: {mean_type} not implemented.')

        self.scale = torch.ones((self.mean.shape)) * scale if isinstance(scale, float) \
                    else torch.tensor(scale)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample(self.batch_dims, self.device)

    def sample(self, n_samples, device='cpu'):
        if not isinstance(n_samples, int):
            n_samples = n_samples[0]
        ks = np.arange(self.mean.shape[0])
        k = np.random.choice(a=ks, size=n_samples)
        mean = self.mean[k].to(device)
        scale = self.scale[k].to(device)

        tangent_vec = self.manifold.random_normal_tangent(
            base_point=mean, n_samples=n_samples)
        tangent_vec = tangent_vec * scale

        samples = self.manifold.exp(tangent_vec, mean)
        return samples

    def log_prob(self, samples):
        
        def component_log_prob(mean, scale):
            dist = Wrapped(scale, self.batch_dims, self.manifold, 'mixture', mean=mean)
            return dist.log_prob(samples)

        device = samples.device
        K = self.mean.shape[0]
        means = self.mean.to(device)
        scales = self.scale.to(device)
        component_log_like = []
        for mean, scale in zip(means, scales):
            component_log_like.append(component_log_prob(mean, scale))
        component_log_like = np.stack(component_log_like, axis=0)
        b = 1 / K * np.ones_like(component_log_like)
        return logsumexp(component_log_like, axis=0, b=b)