from collections import namedtuple
import os
import torch
import numpy as np
import random

TrainState = namedtuple(
    "TrainState",
    [
        "optimizerf",
        "schedulerf",
        "modelf",
        "emaf",
        "optimizerb",
        "schedulerb",
        "modelb",
        "emab",
        "step"
    ],
)


def restore_ckpt(ckpt_dir, state, device):
    loaded_state = torch.load(os.path.join(ckpt_dir, 'state.pth'), map_location=device)
    state.optimizerf.load_state_dict(loaded_state["optimizerf"])
    state.schedulerf.load_state_dict(loaded_state["schedulerf"])
    state.modelf.load_state_dict(loaded_state["modelf"], strict=False)
    state.emaf.load_state_dict(loaded_state["emaf"])
    state.optimizerb.load_state_dict(loaded_state["optimizerb"])
    state.schedulerb.load_state_dict(loaded_state["schedulerb"])
    state.modelb.load_state_dict(loaded_state["modelb"], strict=False)
    state.emab.load_state_dict(loaded_state["emab"])
    state = state._replace(step=loaded_state["step"])
    return state


def save_ckpt(ckpt_dir, state):
    saved_state = {
        'optimizerf': state.optimizerf.state_dict(),
        'schedulerf': state.schedulerf.state_dict(),
        'modelf': state.modelf.state_dict(),
        'emaf': state.emaf.state_dict(),
        'optimizerb': state.optimizerb.state_dict(),
        'schedulerb': state.schedulerb.state_dict(),
        'modelb': state.modelb.state_dict(),
        'emab': state.emab.state_dict(),
        'step': state.step
    }
    torch.save(saved_state, os.path.join(ckpt_dir, 'state.pth'))


def set_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed