import os
import socket
import logging
from timeit import default_timer as timer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm # For terminal print
from copy import deepcopy
import gc

import torch
import numpy as np
import math

from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class
import util.cfg # For Omegaconf customization

from data.tensordataset import TensorDataset, DataLoader, random_split
from losses import get_loss_step_fn
from util.run_utils import TrainState, restore_ckpt, save_ckpt, set_seed
from util.vis import earth_plot, plot_tori, plot_mesh, plot_so3, plot_hyperbolic, ramachandran_plot
from util.loggers_pl import LoggerCollection

log = logging.getLogger(__name__)

def run(cfg):
    def train(train_state, best_val=False):
        best_logp = -200

        loss_fn = instantiate(cfg.loss, mix=mix)
        train_step_fn = get_loss_step_fn(loss_fn, clip_grad_norm=cfg.grad_norm, lr_sched=cfg.lr_sched)

        tbar = tqdm(
            range(train_state.step, cfg.steps),
            total=cfg.steps - train_state.step,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=1,
        )
        train_time = timer()

        total_train_time = 0
        for _ in tbar:
            batch = next(train_ds)
            train_state, lossf, lossb = train_step_fn(train_state, batch.to(device))

            if torch.isnan(lossf+lossb).any():
                log.warning("Loss is nan")
                return train_state, best_logp, False

            step = train_state.step
            if step % 10 == 0:
                logger.log_metrics({"train/loss_f": lossf.item()}, step)
                logger.log_metrics({"train/loss_b": lossb.item()}, step)
                tbar.set_description(f"F: {lossf:.2f} | B: {lossb:.2f}")

            if step % cfg.val_freq == 0:
                logger.log_metrics(
                    {"train/time_per_it": (timer() - train_time) / cfg.val_freq}, step
                )
                total_train_time += timer() - train_time
                eval_time = timer()

                if cfg.train_val:
                    logp = evaluate(train_state, "val", step)
                    logger.log_metrics({"val/time_per_it": (timer() - eval_time)}, step)

                    if best_val:
                        if logp > best_logp:
                            best_logp = logp
                            save_ckpt(ckpt_path, train_state)
                    else:
                        save_ckpt(ckpt_path, train_state)

                    gc.collect()

                    # NOTE: For observation
                    if best_val and step % (cfg.val_freq * 10) == 0:
                        saved_state = restore_ckpt(ckpt_path, deepcopy(train_state), device)
                        evaluate(saved_state, "test", saved_state.step, best_logp=best_logp)

                        if step > saved_state.step + cfg.patience:
                            return train_state, best_logp, True

                if cfg.train_plot and step % cfg.plot_freq == 0:
                    generate_plots(train_state, "val", step=step)
                train_time = timer()

        logger.log_metrics({"train/total_time": total_train_time}, step)
        return train_state, best_logp, True


    def evaluate(train_state, stage, step, **kwargs):
        try:
            dataset = eval_ds if stage == "val" else test_ds

            modelf = train_state.modelf
            modelb = train_state.modelb

            emaf = train_state.emaf
            emab = train_state.emab
            emaf.copy_to(modelf.parameters())
            emab.copy_to(modelb.parameters())

            likelihood_fn = likelihood.get_log_prob(modelf, modelb)

            logp, nfe, N = 0.0, 0.0, 0
            tot = 0
            if hasattr(dataset, "__len__"):
                for batch in dataset:
                    if len(batch)>0:
                        logp_step, nfe_step = likelihood_fn(batch.to(device))
                        logp += logp_step.sum()
                        nfe += nfe_step
                        N += logp_step.shape[0]
            else:
                dataset.batch_dims = cfg.eval_batch_size
                num_rounds = round(20_000 / cfg.eval_batch_size)
                for i in range(num_rounds):
                    batch = next(dataset)
                    logp_step, nfe_step = likelihood_fn(batch.to(device))
                    logp += logp_step.sum()
                    nfe += nfe_step
                    N += logp_step.shape[0]
                    tot += logp_step.shape[0]
                dataset.batch_dims = cfg.batch_size

            logp /= N
            nfe /= len(dataset) if hasattr(dataset, "__len__") else num_rounds

            logger.log_metrics({f"{stage}/logp": logp}, step)
            logger.log_metrics({f"{stage}/nfe": nfe}, step)

            with logging_redirect_tqdm():
                if stage == "test" and cfg.best_val:
                    log.info(f">>> [Epoch {step:06d}] | Val logp={kwargs['best_logp']:.3f} | "
                                f"Test logp={logp:.3f} | nfe: {nfe:.1f}")
                else:
                    log.info(f"[Epoch {step:06d}] {stage} logp: {logp:.3f} | nfe: {nfe:.1f}")
            logger.save()

            return logp
        except:
            return -10000


    def generate_plots(train_state, stage, step=None):
        try:
            modelf = train_state.modelf
            modelb = train_state.modelb

            emaf = train_state.emaf
            emab = train_state.emab
            emaf.copy_to(modelf.parameters())
            emab.copy_to(modelb.parameters())

            fdrift_fn = mix.get_drift_fn(modelf)
            bdrift_fn = mix.rev().get_drift_fn(modelb)
            sde = mix.approx(fdrift_fn, bdrift_fn, cfg.use_pode)

            likelihood_fn = likelihood.get_log_prob(modelf, modelb)
            log_prob = lambda x: likelihood_fn(x)[0]

            if cfg.name == 'so3':
                plot_args = {'N': 50, 'surf_cnt': 15, 'pmax': 12.0, 'pmin': -5.0}
                plt = plot_so3(test_ds, log_prob, cfg.data_dir, **plot_args)
            elif cfg.name == 'hyperbolic':
                plt = plot_hyperbolic(test_ds, log_prob)
            elif cfg.name in ['general', 'glycine', 'proline', 'prepro']:
                plt = ramachandran_plot(test_ds, log_prob, device=device)
            else:
                NUM_SAMPLES = 2**14
                shape = (cfg.sample_batch_size,) #(cfg.batch_size,)
                sampler = instantiate(cfg.sampler, sde=sde, shape=shape, 
                                        N=1000, eps=cfg.eps, device=device)

                num_rounds = math.ceil(NUM_SAMPLES / shape[0])
                samples = []
                for i in tqdm(range(num_rounds), position=1, leave=False):
                    samples.append(sampler(prior_samples=None))
                samples = torch.cat(samples, dim=0)

                prop_in_M = manifold.belongs(samples, atol=1e-4).sum() / samples.shape[0]
                if prop_in_M < 0.999:
                    log.info(f"Prop samples in M = {100 * prop_in_M.item():.1f}%")

                if cfg.name in ['flood', 'fire', 'earthquake', 'volcano']:
                    logp = log_prob(samples)
                    plt = earth_plot(cfg.dataset.name, train_ds, test_ds, samples, logp)
                elif cfg.name == 'rna':
                    data_samples = train_ds.dataset.dataset.data
                    train_dix = train_ds.dataset.indices
                    plt = plot_tori(data_samples[train_dix], samples)
                elif cfg.name == 'htori':
                    data_samples = eval_ds.sample(NUM_SAMPLES)
                    plt = plot_tori(data_samples, samples)
                elif cfg.name in ['bunny', 'spot']:
                    log_dir = f'logs/version_{logger.version}'
                    save_path = os.path.join(*[run_path, log_dir, 'images'])
                    
                    logprobs = []
                    for mv in tqdm(torch.split(manifold.vt, 10000), position=1, leave=False):
                        logprobs.append(log_prob(mv))
                    logprobs = np.concatenate(logprobs, axis=0)
                    logp = np.exp(logprobs)

                    plt = plot_mesh(cfg.name, 
                        manifold.vn, manifold.fn, 
                        samples, logp,
                        save_path, step
                    )
                else:
                    raise NotImplementedError(f'Exp: {cfg.name} plot not implemented.')

            if plt is not None:
                logger.log_plot(f"", plt, step)
        except:
            pass

    
    ### Main
    log.info(cfg)
    if torch.cuda.is_available():
        # NOTE: Multi-gpu not enabled due to torch.vmap
        device = 'cuda:0'
    else:
        device = 'cpu'

    log.info(f"Torch devices: {device}")
    log.info("Stage : Start")
    run_path = os.getcwd()
    # log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)

    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    log.info(f"SEED: {cfg.seed}")
    set_seed(cfg.seed)

    log.info("Stage : Instantiate dataset")
    dataset = instantiate(cfg.dataset)

    if isinstance(dataset, TensorDataset):
    # split and wrapp dataset into dataloaders
        if cfg.name in ['volcano', 'earthquake', 'flood', 'fire', 'spot', 'bunny']:
            train_ds, eval_ds, test_ds = random_split(dataset, lengths=cfg.splits)

        elif cfg.name in ['general', 'glycine', 'proline', 'prepro', 'rna']:
            N = len(dataset)
            N_val = N_test = N // 10
            N_train = N - N_val - N_test
            train_ds, eval_ds, test_ds = torch.utils.data.random_split(
                dataset,
                [N_train, N_val, N_test],
                generator=torch.Generator().manual_seed(cfg.seed),
            )
        else:
            raise NotImplementedError(f'Exp: {cfg.name} not implemented.')

        train_ds, eval_ds, test_ds = (
            DataLoader(train_ds, batch_dims=cfg.batch_size, shuffle=True),
            DataLoader(eval_ds, batch_dims=cfg.eval_batch_size),
            DataLoader(test_ds, batch_dims=cfg.eval_batch_size),
        )
        log.info(
            f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
        )

    else:
        dataset.device = device
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    manifold = dataset.manifold

    log.info("Stage : Instantiate mixture")
    beta_schedule = instantiate(cfg.beta_schedule)
    mix = instantiate(cfg.mix, manifold=manifold, beta_schedule=beta_schedule)
    likelihood = instantiate(cfg.likelihood, mix=mix)

    log.info("Stage : Instantiate model / optimizer")

    modelf_cfg = cfg.get('model', cfg.modelf)
    modelb_cfg = cfg.get('model', cfg.modelb)

    modelf = instantiate(modelf_cfg, manifold=manifold).to(device)
    modelb = instantiate(modelb_cfg, manifold=manifold).to(device)

    emaf = instantiate(cfg.ema, parameters=modelf.parameters())
    emab = instantiate(cfg.ema, parameters=modelb.parameters())

    optimizerf = instantiate(cfg.optim, params=modelf.parameters())
    optimizerb = instantiate(cfg.optim, params=modelb.parameters())

    schedulerf = instantiate(cfg.scheduler, optimizer=optimizerf)
    schedulerb = instantiate(cfg.scheduler, optimizer=optimizerb)

    # NOTE: state contains actual objects of models, optimizers, and emas, 
    # not just the parameters
    train_state = TrainState(
        optimizerf=optimizerf,
        schedulerf=schedulerf,
        modelf=modelf,
        emaf=emaf,
        optimizerb=optimizerb,
        schedulerb=schedulerb,
        modelb=modelb,
        emab=emab,
        step=0
    )

    if cfg.resume or cfg.mode == 'test':
        train_state = restore_ckpt(ckpt_path, train_state, device)
        best_logp = -200.0
    else:
        save_ckpt(ckpt_path, train_state)

    if cfg.mode == "train" or cfg.mode == "all":
        if train_state.step == 0 and cfg.test_plot:
            # generate_plots(train_state, "test", step=0)
            pass
        log.info("Stage : Training")
        train_state, best_logp, success = train(train_state, cfg.best_val)

    if cfg.mode == "test" or (cfg.mode == "all" and success):
        train_state = restore_ckpt(ckpt_path, train_state, device)

        log.info("Stage : Test")
        if cfg.test_test:
            evaluate(train_state, "test", step=train_state.step, best_logp=best_logp)
        if cfg.test_plot:
            generate_plots(train_state, "test", step=train_state.step)
        success = True
    logger.save()
    logger.finalize("success" if success else "failure")