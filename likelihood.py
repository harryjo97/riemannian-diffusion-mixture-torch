import torch
import numpy as np
from torchdiffeq import odeint
from util.div_utils import get_pode_drift, get_riemannian_div_fn, div_noise

class Flow:  # NOTE: Flow: data -> base
    def __init__(self, mix, eps=1.0e-3, rtol=1.0e-5, atol=1.0e-5, method='dopri5', exact=True):
        self.mix = mix
        self.eps = eps
        self.test_ode_kwargs = dict(rtol=rtol, atol=atol, method=method)
        assert method in ['dopri5', 'euler']
        self.adaptive = (method=='dopri5')
        self.exact = exact

    def get_forward(self, modelf, modelb, data):
        with torch.no_grad():
            shape = data.shape
            t_start = self.mix.t0 + self.eps if self.mix.pred else self.mix.t0
            t_end = self.mix.tf-self.eps
            if self.adaptive:
                ts = torch.tensor([t_start, t_end], device=data.device) 
            else:
                ts = torch.linspace(t_start, t_end, 1001, device=data.device)
            self.nfe_counter = 0
            noise_type = 'None' if self.exact else 'Rademacher'

            drift_fn = get_pode_drift(modelf=modelf, modelb=modelb, mix=self.mix)
            div_fn = get_riemannian_div_fn(drift_fn, noise_type, self.mix.manifold)

            # Solving for the change in log-likelihood
            def ode_func(t, y):
                self.nfe_counter += 1
                vec_t = torch.ones((y.shape[0],), device=y.device) * t
                sample = y[:, :-1]
                
                drift = drift_fn(sample, vec_t)
                epsilon = None if self.exact else div_noise(sample.shape, noise_type, y.device)
                logp_grad = div_fn(sample, vec_t, epsilon).reshape([shape[0], 1])
                return torch.cat([drift, logp_grad], dim=1)

            data = data.reshape(shape[0], -1)
            init = torch.cat([data, torch.zeros((shape[0], 1), device=data.device)], dim=1)
            if self.adaptive:
                solution = odeint(ode_func, init, ts, **self.test_ode_kwargs)
                z = solution[-1, ..., :-1]
                delta_logp = solution[-1, ..., -1].detach().cpu().numpy()
            else:
                solution = projx_integrator_return_last(self.mix.manifold, ode_func, init, ts)
                z = solution[:, :-1]
                delta_logp = solution[:, -1].detach().cpu().numpy()
        return z, delta_logp, self.nfe_counter


class Likelihood:
    def __init__(self, mix, rtol=1.0e-5, atol=1.0e-5, method='dopri5', exact=True):
        self.mix = mix
        self.base = mix.prior
        self.flow = Flow(mix=mix, rtol=rtol, atol=atol, method=method, exact=exact)

    def get_log_prob(self, modelf, modelb):
        def log_prob(x):
            z, inv_logdets, nfe = self.flow.get_forward(modelf, modelb, x)
            log_prob = self.base.log_prob(z).reshape(-1)
            log_prob += inv_logdets
            return np.clip(log_prob, -1e38, 1e38), nfe
        return log_prob

from tqdm import tqdm

@torch.no_grad()
def projx_integrator_return_last(
    manifold, odefunc, x0, t
):
    """Has a lower memory cost since this doesn't store intermediate values."""
    xt = x0
    t0s = t[:-1]
    t0s = tqdm(t0s, desc='logp', position=1, leave=False)

    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt)
        xt = xt + dt * vt
        xt = torch.cat([manifold.projection(xt[:, :-1]), xt[:,-1].reshape(-1, 1)], dim=1)
    return xt