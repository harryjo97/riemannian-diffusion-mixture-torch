# Riemannian Diffusion Mixture

This repo contains a PyTorch implementation for the paper [Generative Modeling on Manifolds Through Mixture of Riemannian Diffusion Processes](https://arxiv.org/abs/2310.07216). 

We provide official code repo for JAX implementation in [riemannian-diffusion-mixture-torch](https://github.com/harryjo97/riemannian-diffusion-mixture-torch).

## Why Riemannian Diffusion Mixture?

- Simple design of the generative process as a mixture of Riemannian bridge processes, which does not require heat kernel estimation as previous denoising approach.
- Geometrical interpretation for the mixture process as the weighted mean of tangent directions on manifolds
- Scales to higher dimensions with significantly faster training compared to previous diffusion models.


## Dependencies

Create an environment with Python 3.9.0, and Pytorch 2.0.0. Install requirements with the following command:
```
pip install -r requirements.txt
conda install -c conda-forge cartopy python-kaleido
```

## Manifolds

Following manifolds are supported in this repo:
- Euclidean
- Hypersphere
- Torus
- Hyperboloid
- Triangular mesh
- Special orthogonal group

To implement new manifolds, add python files that define the geometry of the manifold in `/geomstats/geometry`.

Please refer to [geomstats/geometry](https://github.com/geomstats/geomstats/tree/main/geomstats/geometry) for examples.

## Running Experiments

This repo supports experiments on the following datasets:
- Protein datasets: `General`, `Glycine`, `Proline`, and `Pre-Pro`, and `RNA`.
- High-dimensional tori

Please refer to [riemannian-diffusion-mixture](https://github.com/harryjo97/riemannian-diffusion-mixture) for running expreiments on `earth and climate science datasets`, `triangular mesh datasets`, and `hyperboloid datasets`.

### 1. Dataset preparations

For experiment on Protein datasets, create .tsv file in `/data/top500` directory with the following command:
```sh
cd data/top500
bash batch_download.sh -f list_file.txt -p
python get_torsion_angle.py
```

For experiment on RNA dataset, create .tsv file in `/data/rna` directory with the following command:
```sh
cd data/rna
bash batch_download.sh -f list_file.txt -p
python get_torsion_angles.py
```

### 2. Configurations

The configurations are provided in the `config/` directory in `YAML` format. 

### 3. Experiments

```
CUDA_VISIBLE_DEVICES=0 python main.py -m \
    experiment=<exp> \
    seed=0,1,2,3,4 \
    n_jobs=5 \
```
where ```<exp>``` is one of the experiments in `config/experiment/*.yaml`

For example,
```
CUDA_VISIBLE_DEVICES=0 python main.py -m \
    experiment=rna \
    seed=0,1,2,3,4 \
    n_jobs=5 \
```

To run experiments on high-dimensional tori, use `experiment=htori` with `n=$DIM` where `$DIM` denotes the dimesion of the tori.

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@inproceedings{jo2024riemannian,
  author    = {Jaehyeong Jo and
               Sung Ju Hwang},
  title     = {Generative Modeling on Manifolds Through Mixture of Riemannian Diffusion Processes},
  booktitle = {International Conference on Machine Learning},
  year      = {2024},
}
```

## Acknowledgments

Our code builds upon [geomstats](https://github.com/geomstats/geomstats). We thank [Riemannian Score-Based Generative Modelling](https://github.com/oxcsml/riemannian-score-sde?tab=readme-ov-file) and [Riemmanian Flow Matching](https://github.com/facebookresearch/riemannian-fm) for their works.