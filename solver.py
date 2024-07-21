import abc
import torch
import numpy as np
from tqdm import trange
from torchdiffeq import odeint

from util.registry import register_category

get_predictor, register_predictor = register_category("predictors")
get_corrector, register_corrector = register_category("correctors")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde):
        super().__init__()
        self.sde = sde
        self.manifold = sde.manifold

    @abc.abstractmethod
    def update_fn(self, x, t, dt):
        """One update of the predictor.
        """
        raise NotImplementedError()


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x0, x, t):
        """One update of the corrector.
        """
        raise NotImplementedError()


# -------- GRW --------
@register_predictor
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, x, t, dt):
        shape = x.shape
        z = self.sde.manifold.random_normal_tangent(
            base_point=x, n_samples=shape[0]).reshape(shape[0], -1)
        drift, diffusion = self.sde.coefficients(x, t)

        if isinstance(dt, torch.Tensor):
            tangent_vector = torch.einsum("...,...i,...->...i", diffusion, z, dt.abs().sqrt())
            tangent_vector = tangent_vector + torch.einsum("...i,...->...i", drift, dt)
        else:
            tangent_vector = torch.einsum("...,...i->...i", diffusion, z) * np.sqrt(np.abs(dt))
            tangent_vector = tangent_vector + drift * dt
        x = self.manifold.exp(tangent_vec=tangent_vector, base_point=x)
        return x, x


@register_corrector
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, snr, n_steps):
        pass

    def update_fn(self, x0, x, t):
        return x, x


class LangevinCorrector(Corrector):
    def __init__(self, sde, snr, n_steps):
        super().__init__(sde, snr, n_steps)
        self.manifold = self.sde.manifold

    def update_fn(self, x0, x, t):
        for i in range(self.n_steps):
            grad = self.score_fn(x0, x, t)
            noise = self.manifold.random_normal_tangent(
                    base_point=x, n_samples=x.shape[0]).reshape(x.shape[0], -1)
            step_size = (self.snr * self.norm(noise) / self.norm(grad)) ** 2 * 2 * torch.ones_like(t)
            tangent_vec = torch.einsum("...i,...->...i", noise, (step_size * 2).sqrt())
            tangent_vec = tangent_vec + torch.einsum("...i,...->...i", grad, step_size)
            x = self.manifold.exp(tangent_vec=tangent_vec, base_point=x)
        return x, x

    def score_fn(self, x0, x, t):
        fdrift, fdiff = self.sde.coefficients(x, t)
        beta_t = fdiff.square()
        # TODO: drift_before_scale with destination x0
        bbridge = self.get_bbridge(x0)
        bdrift = bbridge.drift_before_scale(x, t)
        score = fdrift + torch.einsum("...i,...->...i", bdrift, self.sde.time_scale(t))
        return torch.einsum("...i,...->...i", score, 1./beta_t)

    def norm(self, z):
        norm = lambda y: self.manifold.metric.squared_norm(y)
        return norm(z.reshape(z.shape[0], -1)).sqrt().mean()

    def get_bbridge(self, x0):        
        from manifold.mesh import Mesh
        from sde_lib import SpectralBridge
        if isinstance(self.manifold, Mesh):
            #TODO: wtype could be other than inv
            params={'manifold':self.manifold, 'beta_schedule':self.sde.beta_schedule, 
                    'dest':x0, 'drift_scale':0., 'wtype':'inv'}
            bbridge = SpectralBridge(**params)
        else:
            raise NotImplementedError(f'Manifold is not a mesh.')
        return bbridge


def get_pc_sampler(sde, shape, N, 
                    predictor="EulerMaruyamaPredictor",
                    corrector="NoneCorrector",
                    snr=0.1, n_steps=1,
                    eps=1.0e-3, device='cpu'): 
    """Create a Predictor-Corrector (PC) sampler.
    """
    assert sde.approx
    predictor = get_predictor(predictor if predictor is not None else "EulerMaruyamaPredictor")(
        sde
    )
    corrector = get_corrector(corrector if corrector is not None else "NoneCorrector")(
        sde, snr, n_steps
    )

    def pc_sampler(prior_samples=None):
        with torch.no_grad():
            # Initial samples
            x = prior_samples if prior_samples is not None else \
                sde.prior_sampling(shape, device).reshape(shape[0], -1)
            x0 = x

            timesteps = torch.linspace(sde.t0, sde.tf-eps, N, device=x.device)
            dt = (timesteps[-1] - timesteps[0]) / N

            # Diffusion process 
            tbar = tqdm(range(0, N), position=1, leave=False)
            for i in tbar:
                vec_t = torch.ones((x.shape[0],), device=x.device) * timesteps[i]
                x, x_mean = corrector.update_fn(x0, x, vec_t)
                x, x_mean = predictor.update_fn(x, vec_t, dt)

        return x_mean

    return pc_sampler


def get_pode_sampler(sde, shape, eps=1.0e-3, device='cpu', **kwargs): 
    """Create a probability flow ODE sampler.
    """
    assert sde.approx
    ode_kwargs = dict(rtol=1.0e-5, atol=1.0e-5)

    def ode_func(t, y):
        y = sde.manifold.projection(y)
        vec_t = torch.ones((y.shape[0],), device=y.device) * t
        drift = sde.drift(y, vec_t)
        drift = sde.manifold.to_tangent(vector=drift, base_point=y)
        return drift

    def pode_sampler(prior_samples=None):
        """The PC sampler funciton.
        """
        with torch.no_grad():
            # Initial samples
            init = prior_samples if prior_samples is not None else \
                sde.prior_sampling(shape, device)
            ts = torch.tensor([sde.t0, sde.tf-eps], device=device)
            solution = odeint(ode_func, init, ts, **ode_kwargs)
            x = solution[-1]
        # Projection to the manifold
        return sde.manifold.projection(x)

    return pode_sampler


def get_grw_sampler(sde, N=100, 
                    eps=1.0e-3,): 
    """Create a GRW sampler.
    """
    predictor = get_predictor(predictor if predictor is not None else "EulerMaruyamaPredictor")(
        sde
    )

    def sampler(x, t0=None, tf=None):
        """The GRW sampler funciton.
        """
        with torch.no_grad():
            t0 = sde.t0 if t0 is None else t0
            tf = sde.tf if tf is None else tf
            
            # Called only for marginal sample xt starting from x0
            tf = tf - eps
            stf = tf.detach().cpu() if isinstance(tf, torch.Tensor) else tf

            timesteps = np.linspace(t0, stf, N)
            timesteps = torch.from_numpy(timesteps).to(x.device)
            dt = (tf - t0) / N
            # -------- GRW --------
            for i in range(0, N-1): # Exclude the final result do to numerical instability for N=100
                t = timesteps[i]
                x, x_mean = predictor.update_fn(x, t, dt)
            return x_mean
    return sampler


class EulerMaruyamaTwoWayPredictor:
    def __init__(self, mix, x0, xf, mask):
        self.mix = mix
        self.x0 = x0
        self.xf = xf
        self.mask = mask

        self.manifold = mix.manifold
        self.fsde = mix.bridge(xf)
        self.bsde = mix.rev().bridge(x0)

    def update_fn(self, x, t, dt):
        shape = x.shape
        z = self.manifold.random_normal_tangent(
            base_point=x, n_samples=shape[0]).reshape(shape[0], -1)
        fdrift, fdiff = self.fsde.coefficients(x, t)
        bdrift, bdiff = self.bsde.coefficients(x, t)

        drift = torch.einsum("...i,...->...i", fdrift, self.mask) + \
                torch.einsum("...i,...->...i", bdrift, ~self.mask)
        diffusion = fdiff * self.mask + bdiff * ~self.mask

        tangent_vector = torch.einsum("...,...i,...->...i", diffusion, z, dt.abs().sqrt())
        tangent_vector = tangent_vector + torch.einsum("...i,...->...i", drift, dt)
        x = self.manifold.exp(tangent_vec=tangent_vector, base_point=x)
        return x, x


def get_twoway_sampler(mix, N=10): 
    """Create a Two-way sampler.
    """
    def sampler(x0, xf, t):
        with torch.no_grad():
            t_mask = t < 0.5
            predictor = EulerMaruyamaTwoWayPredictor(mix, x0, xf, t_mask)
            x = torch.einsum("...i,...->...i", x0, t_mask) + \
                torch.einsum("...i,...->...i", xf, ~t_mask)

            ts = t * t_mask + (1.-t) * ~t_mask
            timesteps = np.linspace(mix.t0, ts.detach().cpu(), N)
            timesteps = torch.from_numpy(timesteps).to(x.device)
            dt = (ts - mix.t0) / N
            # -------- GRW --------
            for i in range(0, N):
                t = timesteps[i]
                x, x_mean = predictor.update_fn(x, t, dt)
        return x_mean
    return sampler
