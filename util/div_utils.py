import torch
import numpy as np

def div_noise(shape, hutchinson_type, device):
    """Sample noise for the hutchinson estimator."""
    if hutchinson_type == "Gaussian":
        epsilon = torch.randn(shape, device=device)
    elif hutchinson_type == "Rademacher":
        epsilon = torch.randint(low=0, high=2, size=shape, device=device)
        epsilon = epsilon * 2 - 1
    elif hutchinson_type == "None":
        epsilon = None
    else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
    return epsilon


def get_estimate_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""
    def div_fn(y, t, eps):
        grad_fn = lambda z: torch.sum(fn(z, t) * eps)
        grad_fn_eps = torch.func.grad(grad_fn)(y)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(y.shape))))
    return div_fn


def get_exact_div_fn(fn):
    "flatten all but the last axis and compute the true divergence"
    def div_fn(y, t):
        dim = np.prod(y.shape[1:])
        jac = torch.vmap(torch.func.jacrev(fn, argnums=0))(y, t)
        jac = jac.reshape(-1, dim, dim)
        return torch.vmap(torch.trace)(jac)
    return div_fn


def get_pode_drift(mix, modelf, modelb):
    def drift_fn(y, t):
        """The drift function of the probability flow ODE."""
        pode = mix.probability_ode(modelf, modelb)
        y = mix.manifold.projection(y)
        drift = pode.coefficients(y, t)[0]
        # NOTE: Drift should be tangent
        return mix.manifold.to_tangent(drift, y)
    return drift_fn


def get_div_fn(drift_fn, hutchinson_type=False):
    """Euclidean divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda y, t, eps: get_exact_div_fn(drift_fn)(y, t)
    else:
        return lambda y, t, eps: get_estimate_div_fn(drift_fn)(y, t, eps)


def get_riemannian_div_fn(func, hutchinson_type, manifold):
    """divergence of the drift function.
    if M is submersion with euclidean ambient metric: div = div_E
    else (in a char) div f = 1/sqrt(g) \sum_i \partial_i(sqrt(g) f_i)
    """
    sqrt_g = (
        lambda x: 1.0
        if manifold is None or not hasattr(manifold.metric, "lambda_x")
        else manifold.metric.lambda_x(x)
    )
    drift_fn = lambda y, t: sqrt_g(y) * func(y, t)
    div_fn = get_div_fn(drift_fn, hutchinson_type)
    return lambda y, t, eps: div_fn(y, t, eps) / sqrt_g(y)


def tensor_norm(z):
    return torch.norm(z.reshape(z.shape[0], -1), dim=-1).mean()