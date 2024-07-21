import os
import hydra
import torch
from run import run

@hydra.main(config_path="config", config_name="main")
def main(cfg):
    os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    eval('setattr(torch.backends.cudnn, "deterministic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')
    torch.set_default_dtype(torch.float32) 

    return run(cfg)

if __name__ == "__main__":
    main()
