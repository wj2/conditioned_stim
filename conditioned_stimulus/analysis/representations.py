import matplotlib.pyplot as plt
import re
import numpy as np
import sklearn.svm as skm
import sklearn.metrics.pairwise as skmp

import general.neural_analysis as na
import general.plotting as gpl
import general.utility as u


def equals_one(x):
    return x == 1


def equal_0(x):
    return x == 0


default_funcs = {
    "rewarded": equal_0,
}

default_dec_variables = (
    "valence",
    "intensity",
)


def get_bhv_rep_dec_info(
    data,
    t_start,
    sess_ind=0,
    binsize=500,
    binstep=500,
    marker_regex=".*manual.*(x|y)",
    time_zero_field="CS On",
):
    pops, xs = data.get_neural_activity(
        binsize,
        t_start,
        t_start,
        binstep,
        time_zero_field=time_zero_field,
    )
    rep = pops[sess_ind]
    valence = data["valence"][sess_ind].to_numpy()

    markers = list(x for x in data.session_keys if re.match(marker_regex, x))

    marks, xs = data.get_field_timeseries(
        markers, time_zero_field=time_zero_field, begin=t_start, end=t_start + binsize
    )
    bhv = np.nanmean(marks[sess_ind], axis=-1, keepdims=True)
    return rep, bhv, valence


def decode_bhv_rep_corr(rep, bhv, target, mask=None, n_folds=100, rng_seed=None):
    if rng_seed is None:
        rng = np.random.default_rng()
        rng_seed = rng.integers(2**32 - 1)
    if mask is None:
        mask = np.ones_like(target, dtype=bool)

    out_rep = na.fold_skl_shape(
        rep[mask], target[mask], n_folds, rng_seed=rng_seed, return_projection=True
    )
    out_bhv = na.fold_skl_shape(
        bhv[mask], target[mask], n_folds, rng_seed=rng_seed, return_projection=True
    )
    return out_rep, out_bhv


@gpl.ax_adder()
def plot_bhv_rep_corr(
    rep_dict,
    bhv_dict,
    color_info=None,
    ax=None,
    scatter_cmap="Blues",
    bhv_correct=False,
    target=None,
):
    rep_projs = rep_dict["projection"].flatten()
    bhv_projs = bhv_dict["projection"].flatten()
    if color_info is not None:
        inds = rep_dict["test_inds"].flatten()
        colors = color_info[inds]
    else:
        colors = None
    cmap = plt.get_cmap(scatter_cmap)
    if target is not None and bhv_correct:
        bhv_mask = target[inds] == (bhv_projs > 0)
        rep_projs = rep_projs[bhv_mask]
        bhv_projs = bhv_projs[bhv_mask]
        if colors is not None:
            colors = colors[bhv_mask]

    ax.scatter(rep_projs, bhv_projs, c=colors, cmap=cmap)
    ax.set_xlabel("projection in neural space")
    ax.set_ylabel("projection in bhv space")
    gpl.clean_plot(ax, 0)
    print("correlation: ", np.corrcoef(rep_projs, bhv_projs)[1, 0])
    print("rep decoding: ", rep_dict["score"])
    print("bhv decoding: ", bhv_dict["score"])


def get_rep_info(
    data,
    t_start,
    sess_ind=0,
    binsize=500,
    binstep=500,
    time_zero_field="CS On",
    **kwargs,
):
    tnum = data["Trial"][sess_ind]
    block = data["Block"][sess_ind]
    stim = data["fractal_chosen"][sess_ind]
    valence = data["valence"][sess_ind]

    pops, xs = data.get_neural_activity(
        binsize,
        t_start,
        t_start,
        binstep,
        time_zero_field=time_zero_field,
        skl_axes=True,
        **kwargs,
    )
    rep = pops[sess_ind]
    return rep, block, tnum, stim, valence


def _make_tr_mask(block, train_blocks, n_train_trls):
    inds = np.arange(len(block))
    tr_inds = []
    for b in train_blocks:
        mask = b == block
        b_inds = inds[mask][-n_train_trls:]
        tr_inds.extend(b_inds)
    tr_mask = np.isin(inds, tr_inds)
    return tr_mask


@gpl.ax_adder()
def block_rep_change(
    rep,
    block,
    tnum,
    stim,
    valence,
    n_train_trls=100,
    train_blocks=(1,),
    n_folds=50,
    model=skm.LinearSVC,
    valence_cmap="bwr",
    stim_marker=("X", "o", "*", "D"),
    n_trls_avg=8,
    ax=None,
    arrow_len=0.5,
    v_color=(0.2, 0.5, 0.2),
    m_color=(0.2, 0.9, 0.4),
    **kwargs,
):
    valence = valence.to_numpy()
    mag = np.abs(valence)
    valence_binary = valence > np.mean(valence)
    mag = mag > np.mean(mag)
    dec_targets = np.stack((valence_binary, mag), axis=1)

    (rep,) = na.zscore_tc(rep)
    rep = np.squeeze(rep, axis=1)

    tr_mask = _make_tr_mask(block, train_blocks, n_train_trls)
    tr_rep = rep[:, tr_mask]
    tr_targets = dec_targets[tr_mask]

    out = na.fold_skl_flat(
        tr_rep,
        tr_targets,
        n_folds,
        model=model,
        mean=False,
        pre_pca=None,
        norm=False,
        **kwargs,
    )

    basis = na.get_multioutput_coeffs(out["estimators"], pipeline_ind=-1, orthog=True)
    coeffs = na.get_multioutput_coeffs(out["estimators"], pipeline_ind=-1)
    coeffs_mu = np.mean(coeffs[:, 0], axis=0)
    basis_mu = np.mean(basis[:, 0], axis=0)

    te_mask = np.logical_not(tr_mask)
    u_stim = np.unique(stim)
    u_blocks = np.unique(block)
    v_cmap = plt.get_cmap(valence_cmap)
    rep_proj = (basis_mu @ rep[..., 0]).T
    dim_proj = arrow_len * u.make_unit_vector((basis_mu @ coeffs_mu.T).T)

    ax.arrow(
        0, 0, dim_proj[0, 0], dim_proj[0, 1], color=v_color, width=0.01, label="valence"
    )
    ax.arrow(
        0,
        0,
        dim_proj[1, 0],
        dim_proj[1, 1],
        color=m_color,
        width=0.01,
        label="magnitude",
    )
    valence_rs = valence - np.min(valence) - 0.1
    valence_rs = valence_rs / (np.max(valence_rs) + 0.1)
    for i, s in enumerate(u_stim):
        stim_mask = s == stim
        marker = stim_marker[i]
        mask_s = np.logical_and(te_mask, stim_mask).to_numpy()
        rp_s = rep_proj[mask_s]
        block_s = block[mask_s]
        inds = np.arange(len(rp_s))
        ind_dists = skmp.euclidean_distances(inds[:, None]) <= n_trls_avg
        rp_s_avg = np.zeros_like(rp_s)
        for k in range(len(inds)):
            rp_s_avg[k] = np.mean(rp_s[ind_dists[k]], axis=0)
        ax.plot(*rp_s_avg.T, color=(0.9, 0.9, 0.9))
        for j, b in enumerate(u_blocks):
            b_mask = b == block_s
            color = v_cmap(np.median(valence_rs[mask_s][b_mask]))

            ax.plot(*rp_s_avg[block_s == b].T, color=color)
            ax.scatter(
                *rp_s_avg[block_s == b][[0]].T,
                color=(1, 1, 1),
                edgecolors=color,
                marker=marker,
                zorder=-1,
            )
            ax.scatter(
                *rp_s_avg[block_s == b][[-1]].T,
                color=color,
                marker=marker,
                zorder=-1,
            )
    ax.set_aspect("equal")
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax)


def make_variable_masks(
    data,
    dec_variables=default_dec_variables,
    func_dict=None,
    and_mask=None,
):
    """
    The variables of interest are:
    position (binary and continuous), correct side, view direction, choice
    """
    if func_dict is None:
        func_dict = default_funcs
    masks = {}
    for k, v in dec_variables.items():
        func = func_dict.get(k, equals_one)
        m1 = func(data[v])
        m2 = func(data[v]).rs_not()
        if and_mask is not None:
            m1 = m1.rs_and(and_mask)
            m2 = m2.rs_and(and_mask)
        masks[k] = (m1, m2)
    return masks


def make_magnitude_masks(
    data_use,
    block_key="Block",
    block=None,
    valence_key="valence",
    mag_thr=0.5,
):
    m1 = (data_use[valence_key] > mag_thr).rs_or(data_use[valence_key] < -mag_thr)
    m2 = m1.rs_not()
    if block is not None:
        block_mask = data_use[block_key] == block
        m1 = m1.rs_and(block_mask)
        m2 = m2.rs_and(block_mask)
    return m1, m2


def make_valence_masks(
    data_use,
    block_key="Block",
    block=None,
    valence_key="valence",
    val_thr=0.5,
):
    m1 = data_use[valence_key] > val_thr
    m2 = m1.rs_not()
    if block is not None:
        block_mask = data_use[block_key] == block
        m1 = m1.rs_and(block_mask)
        m2 = m2.rs_and(block_mask)
    return m1, m2


def decode_masks(data, m1, m2, tbeg, tend, winsize=500, winstep=50, **kwargs):
    out = data.decode_masks(m1, m2, winsize, tbeg, tend, winstep, **kwargs)
    return out


time_fields = {
    "CS On": (-500, 1500),
    "Trace End": (-2000, 0),
}
mask_funcs = {
    "magnitude": make_magnitude_masks,
    "valence": make_valence_masks,
}


def decode_fields_times(data, mask_funcs=mask_funcs, time_fields=time_fields, **kwargs):
    out_dict = {}
    for k_var, mf in mask_funcs.items():
        out_dict[k_var] = {}
        m1, m2 = mf(data)
        for k_tf, (tbeg, tend) in time_fields.items():
            out_dict[k_var][k_tf] = decode_masks(
                data, m1, m2, tbeg, tend, time_zero_field=k_tf, **kwargs
            )
    return out_dict


def plot_decoding_dict(out_dict, axs=None, line_labels=None, fwid=3):
    if axs is None:
        n_times = len(list(out_dict.values())[0])
        n_vars = len(out_dict)
        f, axs = plt.subplots(
            n_vars,
            n_times,
            figsize=(fwid * n_times, fwid * n_vars),
            sharey=True,
        )
    for i, (k_var, var_dict) in enumerate(out_dict.items()):
        for j, (time, out) in enumerate(var_dict.items()):
            dec, xs = out[:2]
            if line_labels is None:
                line_labels = ("",) * len(dec)
            for k, dec_k in enumerate(dec):
                gpl.plot_trace_werr(
                    xs,
                    dec_k,
                    confstd=True,
                    ax=axs[i, j],
                    label=line_labels[k],
                )
            gpl.add_hlines(0.5, axs[i, j])
    return axs
