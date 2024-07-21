import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cartopy
import cartopy.crs as ccrs
import pyvista as pv
import warnings; warnings.filterwarnings("ignore", category=UserWarning)

from torch.distributions.multivariate_normal import MultivariateNormal
from geomstats.geometry.special_orthogonal import _SpecialOrthogonal3Vectors
from geomstats.geometry.hyperbolic import Hyperbolic

def latlon_from_cartesian(points):
    r = torch.linalg.norm(points, axis=-1)
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    lat = -torch.arcsin(z / r)
    lon = torch.arctan2(y, x)
    lon = lon - np.pi # Required for visualization: why?
    return torch.cat([lat.unsqueeze(-1), lon.unsqueeze(-1)], axis=-1)

def earth_plot(dataset, train_ds, test_ds, samples=None, logp=None):
    # parameters
    azimuth_dict = {"earthquake": 70, "fire": 50, "flood": 60, "volcano": 170}
    azimuth = azimuth_dict[dataset]
    polar = 30
    projs = ['plate']
    dparams={
        's'    : 0.2,
        'alpha': 0.4,
        'facecolors': 'none',
        # 'linewidths': 0.1
    }
    sparams={
        's'    : 0.2,
        'alpha': 1.0,
        'vmax' : 1.0, #2.0,
        'vmin' : 0.0, #-2.0,
        'cmap' : 'viridis' #'Greens'
    }

    # create figure with earth features
    figs = []
    for i, proj in enumerate(projs):
        fig = plt.figure(figsize=(5, 5), dpi=300)
        if proj == "ortho":
            projection = ccrs.Orthographic(azimuth, polar)
        elif proj == "robinson":
            projection = ccrs.Robinson(central_longitude=0)
        elif proj == "plate":
            projection = ccrs.PlateCarree(central_longitude=0)
        else:
            raise Exception("Invalid proj {}".format(proj))
        ax = fig.add_subplot(1, 1, 1, projection=projection, frameon=True)
        ax.set_global()

        # earth features
        ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor="#e0e0e0")
        colors = sns.color_palette("hls", 8)

        # plot samples
        if samples is not None:
            z = samples.detach().cpu() if isinstance(samples, torch.Tensor) \
                else torch.tensor(samples)
            z = np.array(latlon_from_cartesian(z)) * 180 / np.pi
            points = projection.transform_points(ccrs.Geodetic(), z[:, 1], z[:, 0])
            if logp is not None:
                likelihood = np.exp(logp)
                sc = ax.scatter(points[:, 0], points[:, 1], c=likelihood, **sparams)

                cax = fig.add_axes([ax.get_position().x1+0.01,
                                    ax.get_position().y0,0.02,
                                    ax.get_position().height])
                cbar = plt.colorbar(sc, cax=cax)
                cbar.ax.get_yaxis().set_ticks([sparams['vmin']])
                cbar.ax.text(1.1, sparams['vmax'], f">={int(sparams['vmax'])}", ha='left', va='center')
                cbar.ax.get_yaxis().labelpad = 0
                cbar.ax.set_ylabel('likelihood', rotation=270)
            else:
                ax.scatter(points[:, 0], points[:, 1], c=[colors[3]], **sparams)

        # plot train dataset
        if train_ds is not None:
            train_idx = train_ds.dataset.indices
            z = train_ds.dataset.dataset.data
            z = np.array(latlon_from_cartesian(z)) * 180 / np.pi
            points = projection.transform_points(ccrs.Geodetic(), z[:, 1], z[:, 0])
            ax.scatter(points[train_idx, 0], points[train_idx, 1], c=[colors[5]], **dparams)

        # plot test dataset
        if test_ds is not None:
            test_idx = test_ds.dataset.indices
            z = test_ds.dataset.dataset.data
            z = np.array(latlon_from_cartesian(z)) * 180 / np.pi
            points = projection.transform_points(ccrs.Geodetic(), z[:, 1], z[:, 0])
            ax.scatter(points[test_idx, 0], points[test_idx, 1], c=[colors[0]], **dparams)
            
        figs.append(fig)
        plt.close(fig)
    return figs


def plot_so3(test_ds, log_prob, data_dir, **plot_args):
    N = plot_args['N']
    surf_cnt = plot_args['surf_cnt']
    pmax = plot_args['pmax']
    pmin = plot_args['pmin']

    xs = np.linspace(-np.pi, np.pi, N)
    ys = np.linspace(-np.pi / 2, np.pi / 2, N)
    zs = np.linspace(-np.pi, np.pi, N)
    x, y, z = np.meshgrid(xs, ys, zs)

    cached = os.path.join(data_dir, f'so3_grid_{N}.pkl')
    if os.path.isfile(cached):
        with open(cached, 'rb') as f:
            samples = pickle.load(f)
    else:
        grid = np.stack([x.flatten(), y.flatten(), z.flatten()]).T
        samples = _SpecialOrthogonal3Vectors().matrix_from_tait_bryan_angles(grid)
        samples = samples.reshape(samples.shape[0], -1)
        with open(cached, 'wb') as f:
            pickle.dump(obj=samples, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    samples = samples.float().to(test_ds.device)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.02,
                        specs=[[{"type": "volume"}, {"type": "volume"}]])
    for i, log_prob_fn in enumerate([test_ds.log_prob, log_prob]):
        prob = np.exp(log_prob_fn(samples))
        fig.add_trace(go.Volume(
            x=x.flatten(), 
            y=y.flatten(), 
            z=z.flatten(), 
            value=prob.flatten(), 
            isomin=pmin,
            isomax=pmax,
            opacity=0.1, # needs to be small to see through all surfaces
            surface_count=surf_cnt, # needs to be a large number for good volume rendering
            colorscale='Inferno'
        ), row=1, col=i+1)

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=7, range=[-np.pi, np.pi],),
            yaxis = dict(nticks=5, range=[-np.pi / 2, np.pi / 2],),
            zaxis = dict(nticks=7, range=[-np.pi, np.pi],),),
        width=1900, #800,
        height=900,
        margin=dict(r=10, l=10, b=10, t=10),
    )
    return fig


def proj_t2(x):
    return torch.remainder(
        torch.stack(
            [torch.arctan2(x[..., 0], x[..., 1]), torch.arctan2(x[..., 2], x[..., 3])],
            axis=-1,
        ),
        np.pi * 2,
    )

def plot_t2(x0, xt, size=5):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=False,
        tight_layout=True,
    )

    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        x = proj_t2(x)
        x = x.detach().cpu() if isinstance(x, torch.Tensor) else x
        axes[i].scatter(x[..., 0], x[..., 1], s=0.1)

    for ax in axes:
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 2 * np.pi])
        ax.set_aspect("equal")
    plt.close(fig)
    return fig

def plot_tn(x0, xt, size=5):
    n = xt.shape[-1] if xt is not None else x0.shape[-1]
    n = min(5, n // 4)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(0.6 * size, 0.6 * size * n / 2),
        sharex=False,
        sharey=False,
        tight_layout=True,
        squeeze=False,
    )
    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        for j in range(n):
            x_ = proj_t2(x[..., (4 * j) : (4 * (j + 1))])
            x_ = x_.detach().cpu() if isinstance(x_, torch.Tensor) else x_
            axes[j][i].scatter(x_[..., 0], x_[..., 1], s=0.1)

    axes = [item for sublist in axes for item in sublist]
    for ax in axes:
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 2 * np.pi])
        ax.set_aspect("equal")
    plt.close(fig)
    return fig

def proj_t1(x):
    return torch.remainder(torch.arctan2(x[..., 0], x[..., 1]), 2 * np.pi)

def plot_t1(x0, xt, size=5):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )

    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        x = proj_t1(x)
        x = x.detach().cpu() if isinstance(x, torch.Tensor) else x
        axes[i].scatter(torch.ones_like(x)*3, x, marker="|")

    for ax in axes:
        ax.set_xlim([0, 2 * np.pi])
    plt.close(fig)
    return fig

def plot_tori(x0, xt, size=10):
    dim = x0.shape[-1] // 2
    if dim==1:
        plot = plot_t1
    elif dim==2:
        plot = plot_t2
    else:
        plot = plot_tn

    return plot(x0, xt, size=size)


def plot_mesh(dataset, v, f, samples, prob, save_path, step):
    os.makedirs(save_path, exist_ok=True)
    mesh_params={
        'color': 'white',
        'cmap' : 'coolwarm',
    }
    pts_params={
        'color': 'red',
        'point_size': 4
    }
    plot_params = {'spot': {'position': (2.0,2.0,2.0), 
                                    'angles': [0,150,0], 
                                    'focal': (0.,0.1,-0.1)
                    },
                    'bunny': {'position': (2.5,2.5,2.5), 
                            'angles': [-10,60,0], 
                            'focal': (0.,0.0,0.2)
                    }
    }
    pltp = plot_params[dataset]
    position = pltp['position']
    angles = pltp['angles']
    focal_point = pltp['focal']
    
    pv.start_xvfb()
    axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
    
    def rotate(mesh):
        rot = {'point': axes.origin, 'inplace': False}
        mesh = mesh.rotate_x(angle=angles[0], **rot)
        mesh = mesh.rotate_y(angle=angles[1], **rot)
        mesh = mesh.rotate_z(angle=angles[2], **rot)
        return mesh

    pl = pv.Plotter()
    pl.camera = pv.Camera()
    pl.camera.position = position
    pl.camera.focal_point = focal_point

    pf = np.concatenate([np.array([3]*f.shape[0]).reshape(-1,1), f], axis=-1)
    poly = pv.PolyData(v, pf)
    poly = rotate(poly)
    pscalars = prob #distribution.cfp
    pl.add_mesh(poly, scalars=pscalars, **mesh_params)

    if samples is not None:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        samples = pv.PolyData(samples)
        samples = rotate(samples)
        pl.add_mesh(samples, **pts_params)

    pl.remove_scalar_bar()
    pl.save_graphic(os.path.join(save_path, f"{step}.pdf"))
    pl.close()

    return None


def make_disk_grid(dim, npts, eps=1e-3, device='cpu'):
    poincare_ball = Hyperbolic(dim=dim, default_coords_type="ball")
    R=1.0
    R = R - eps
    bp = torch.linspace(-R, R, npts, device=device)
    xx, yy = torch.meshgrid((bp, bp))
    twodim = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    mask = torch.linalg.norm(twodim, axis=-1) < 1.0 - eps
    lambda_x = poincare_ball.metric.lambda_x(twodim) ** 2 * mask.float()
    volume = (2 * R) ** dim
    return twodim, volume, lambda_x

def plot_hyperbolic(test_ds, log_prob=None, npts=150):
    if test_ds.manifold.coords_type == 'extrinsic': #Hyperboloid
        coord_map = Hyperbolic._ball_to_extrinsic_coordinates
    else: #Poincare ball
        coord_map = lambda x: x
        
    size=10
    device = test_ds.device
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    cmap = sns.cubehelix_palette(
        light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    )
    xs, volume, lambda_x = make_disk_grid(test_ds.manifold.dim, npts, 1e-2, device)
    idx = torch.nonzero(lambda_x).squeeze()
    ys = xs[idx]
    idx = idx.detach().cpu().numpy()
    lambda_x = lambda_x.detach().cpu().numpy()[idx]
    ys = coord_map(ys)
    xs = xs.detach().cpu().numpy()

    for i, log_prob_fn in enumerate([test_ds.log_prob, log_prob]):
        if log_prob_fn is None:
            continue
        prob = np.exp(log_prob_fn(ys))
        # print(f"{prob.min():.4f} | {prob.mean():.4f} | {prob.max():.4f}")
        idx_not_nan = np.nonzero(~np.isnan(prob))[0]
        nb = len(np.nonzero(np.isnan(prob))[0])
        tot = prob.shape[0]
        # print(f"prop nan in prob: {nb / tot * 100:.1f}%")

        measure = np.zeros((npts * npts))
        measure[idx] = prob * lambda_x

        xs = xs.reshape(npts, npts, 2)
        measure = measure.reshape(npts, npts)
        ax[i].pcolormesh(
            xs[:, :, 0],
            xs[:, :, 1],
            measure,
            cmap=cmap,
            linewidth=0,
            rasterized=True,
            shading="gouraud",
        )   
        ax[i].set_xlim([-1.01, 1.01])
        ax[i].set_ylim([-1.01, 1.01])

        ax[i].add_patch(Circle((0, 0), 1.0, color="black", fill=False, linewidth=2, zorder=10))
        ax[i].set_aspect("equal")
        ax[i].axis("off")
    plt.close(fig)
    return fig


#NOTE: For 2D torus proteins
def ramachandran_plot(test_ds, log_prob=None, N=200, device='cpu'):
    zlim = {'General': [-10.5,1.5], 'Glycine': [-12,2], 
            'Proline': [-17.5,2.5], 'Pre-Pro': [-12,0]}
    zmin, zmax = zlim[test_ds.dataset.dataset.amino]

    xs = np.linspace(-np.pi, np.pi, N)
    ys = np.linspace(-np.pi, np.pi, N)
    x, y = np.meshgrid(xs, ys)
    ang_x, ang_y = (x * 180 / np.pi), (y * 180 / np.pi)

    ang_grid = np.stack([x.flatten(), y.flatten()]).T
    coords = np.stack([np.cos(ang_grid[:,0]), np.sin(ang_grid[:,0]), 
                        np.cos(ang_grid[:,1]), np.sin(ang_grid[:,1])], axis=1)
    coords = torch.from_numpy(coords).float().to(device)
    logp = log_prob(coords)

    fig = go.Figure(data =
     go.Contour(
        x=ang_x.flatten(), 
        y=ang_y.flatten(), 
        z=logp,
        zmax=zmax,
        zmin=zmin,
        colorscale='blues',
        colorbar=dict(
            title='log likelihood',
            titleside='right',
            titlefont=dict(
                size=14,
            )
        )
    ))

    tidx = test_ds.dataset.indices
    td = test_ds.dataset.dataset.data[tidx]
    ang = torch.remainder(
        torch.stack(
            [torch.arctan2(td[..., 1], td[..., 0]), torch.arctan2(td[..., 3], td[..., 2])],
            axis=-1,
        ),
        np.pi * 2,)
    ang = torch.where(ang>np.pi, ang-2*np.pi, ang)
    ang = ang * 180 / np.pi

    fig.add_trace(go.Scatter(
        x = ang[:,0],
        y = ang[:,1],
        mode = 'markers',
        marker = dict(
            color = 'rgba(1.0,0,0,1.0)',
            size = 2
        )
    ))

    fig.update_layout(
        xaxis = dict(range=[-180, 180], tick0=-150, dtick=50, nticks=7, title='\u03C6'),
        yaxis = dict(range=[-180, 180], tick0=-150, dtick=50, nticks=7, title='\u03C8'),
        width=800, height=800
    )
    return fig