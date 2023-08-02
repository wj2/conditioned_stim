
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpa
import pandas as pd
import sklearn.decomposition as skd

import general.plotting as gpl


def plot_corr_time(proj, true_val, trial_num, ax=None, plot_err=False, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for proj_i in proj:
        corr_proj = proj_i*true_val
        if plot_err:
            corr_proj = corr_proj > 0
        gpl.plot_scatter_average(trial_num, corr_proj, ax=ax, **kwargs)
    return ax


def plot_dec_dict(dec_dict, axs=None, fwid=3, **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, len(dec_dict),
                              figsize=(len(dec_dict)*fwid, fwid))
    for i, (k, kd) in enumerate(dec_dict.items()):
        axs[i].hist(kd['test_score'], **kwargs)
        tms = kd.get('test_mask_score')
        if tms is not None:
            axs[i].hist(tms, **kwargs)
        axs[i].set_title(k)
        gpl.add_vlines(.5, axs[i])
    return axs


def plot_video(vid, fax=None):
    if fax is None:
        fax = plt.subplots(1, 1)
    f, ax = fax

    img = ax.imshow(vid[0])

    def animate(i):
        return img.set_data(vid[i])

    ani = mpa.FuncAnimation(f, animate, frames=vid.shape[0])
    return ani, ani.to_html5_video()


def project_videos_basis(data, basis, intercepts, key='video_1', val_key='valence',
                         n_time_pts=20,
                         cumsum=False, ax=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_data = data[key]
    types = data[val_key]

    pts_all = []
    trls_all = {}
    for i, t in enumerate(np.unique(types)):
        color = None
        mask = np.logical_and(types == t, ~pd.isna(data[key]))
        pd_t = plot_data[mask]
        
        for j, v in enumerate(pd_t):
            proj = v @ basis + intercepts.T
            if cumsum:
                proj = np.cumsum(proj, axis=0)
            if j == 0:
                label = 'valence = {}'.format(t)
            else:
                label = ''
            l = ax.plot(*proj.T, color=color, label=label, **kwargs)
            color = l[0].get_color()
    ax.legend(frameon=False)


def plot_valence_trajectories(data, key='video_1', val_key='valence', n_time_pts=20,
                              ax=None, cmap='cool'):
    if ax is None:
        f, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_data = data[key]
    types = data[val_key]
    cmap = plt.get_cmap(cmap)

    pts_all = []
    trls_all = {}
    u_types = np.unique(types)
    norm = len(u_types) - 1
    for i, t in enumerate(u_types):
        mask = np.logical_and(types == t, ~pd.isna(data[key]))
        pd_t = plot_data[mask]
        trls_t = list(v for v in pd_t)
        # print(trls_t[0].shape)
        pts_t = np.concatenate(trls_t, axis=0)
        pts_all.append(pts_t)
        trls_all[t] = trls_t

    p = skd.PCA(3)
    p.fit(np.concatenate(pts_all, axis=0))
    for i, (k, trls_k) in enumerate(trls_all.items()):
        print(k, norm)
        color = cmap(i/norm)
        for t in trls_k:
            gpl.plot_highdim_trace(t, p=p, label=k, ax=ax, color=color)


def plot_valence_decoding_tc(*args, ax=None, labels=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if labels is None:
        labels = ('',)*len(args)
    
    for i, (t1, t2) in enumerate(args):
        gpl.plot_scatter_average(t2, t1, ax=ax, label=labels[i], **kwargs)

    gpl.add_hlines(.5, ax)
    ax.set_xlabel('trial number')
    ax.set_ylabel('decoding performance')
