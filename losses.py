import torch
import numpy as np
from solver import get_twoway_sampler

def get_mix_loss_fn(mix, reduce_mean=False, eps=1e-5, num_steps=10, 
                    weight_type='default', sampler_type='twoway'):
    reduce_op = torch.mean if reduce_mean else \
                lambda *args, **kwargs: torch.sum(*args, **kwargs)
    sampler = get_twoway_sampler(mix, num_steps)
    Z = mix.importance_cum_weight(mix.tf-eps, eps)

    def weight_fn(t):
        if weight_type=='default' :
            weight = 1./mix.beta_schedule.beta_t(t)
        elif 'const' in weight_type:
            weight = float(weight_type.split('_')[-1]) * torch.ones_like(t)
        elif weight_type=='importance':
            weight = torch.ones_like(t) * Z
        else:
            raise NotImplementedError(f'{weight_type} not implemented.')
        return weight

    def loss_fn(modelf, modelb, x):
        shape = x.shape
        # Forward (prior->data) drift
        predf_fn = mix.get_drift_fn(modelf, train=True)
        # Backward (data->prior) drift
        predb_fn = mix.get_drift_fn(modelb, train=True)

        if 'importance' in weight_type:
            t = mix.sample_importance_weighted_time((x.shape[0],), eps, x.device)
        else:
            t = torch.rand(shape[0], device=x.device) * (mix.tf - 2*eps) + eps
        x0 = mix.prior.sample(shape, x.device).reshape(shape[0], -1)

        if sampler_type == 'twoway':
            xt = sampler(x0, x, t)
        else:
            raise NotImplementedError(f'Sampler type: {sampler_type} not implemented.')

        # weight
        weight = weight_fn(t)

        # Forward model loss
        lossesf = predf_fn(xt, t) - mix.bridge(x).drift(xt, t)
        lossesf = 0.5 * mix.manifold.metric.squared_norm(lossesf, xt)
        lossesf = weight * lossesf
        lossesf = reduce_op(lossesf.reshape(lossesf.shape[0], -1), dim=-1)

        # Backward model loss
        lossesb = predb_fn(xt, mix.tf-t) - mix.rev().bridge(x0).drift(xt, mix.tf-t)
        lossesb = 0.5 * mix.manifold.metric.squared_norm(lossesb, xt)
        lossesb = weight * lossesb
        lossesb = reduce_op(lossesb.reshape(lossesb.shape[0], -1), dim=-1)

        lossf, lossb = torch.mean(lossesf), torch.mean(lossesb)

        return lossf+lossb, lossf, lossb
    return loss_fn


def get_loss_step_fn(loss_fn, clip_grad_norm=1.0, lr_sched=False):
    """Create a one-step training function.
    """

    def step_fn(state, batch):
        optimizerf, schedulerf, modelf, emaf = state.optimizerf, state.schedulerf, state.modelf, state.emaf
        optimizerb, schedulerb, modelb, emab = state.optimizerb, state.schedulerb, state.modelb, state.emab

        optimizerf.zero_grad()
        optimizerb.zero_grad()

        loss, lossf, lossb = loss_fn(modelf, modelb, batch)
        loss.backward()

        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(modelf.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(modelb.parameters(), clip_grad_norm)

        optimizerf.step()
        optimizerb.step()

        if lr_sched:
            schedulerf.step()
            schedulerb.step()

        # -------- EMA update --------
        emaf.update(modelf.parameters())
        emab.update(modelb.parameters())

        step = state.step + 1
        new_train_state = state._replace(
            step=step,
            optimizerf=optimizerf,
            schedulerf=schedulerf,
            modelf=modelf,
            emaf=emaf,
            optimizerb=optimizerb,
            schedulerb=schedulerb,
            modelb=modelb,
            emab=emab,
        )

        return new_train_state, lossf, lossb
    return step_fn