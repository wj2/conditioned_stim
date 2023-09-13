import numpy as np
import sklearn.svm as skm
import sklearn.model_selection as skms
import sklearn.decomposition as skd
import pandas as pd

import general.neural_analysis as na
import general.utility as u

default_predictors = (
    "sum_lick_count_window",
    "sum_blink_count_window",
    "lick_duration",
    "blink_duration_offscreen",
    "pupil_raster_window_avg",
    "blink_duration_window",
    "eye_distance",
)
history_predictors = (
    "reward_1_back",
    "reward_2_back",
    "reward_3_back",
    "reward_4_back",
    "reward_5_back",
    "airpuff_1_back",
    "airpuff_2_back",
    "airpuff_3_back",
    "airpuff_4_back",
    "airpuff_5_back",
)


def _make_nan_mask(X):
    nan_mask = np.logical_not(np.any(np.isnan(X), axis=1))
    return nan_mask


def _make_block_masks(
    data, train_block, test_block, block_key="block", existing_mask=None
):
    if existing_mask is None:
        existing_mask = np.ones(len(data), dtype=bool)
    train_mask = np.logical_and(existing_mask, data[block_key] == train_block)
    test_mask = np.logical_and(existing_mask, data[block_key] == test_block)
    return train_mask, test_mask


def _make_pos_mask(data, *args, thr=0, existing_mask=None):
    nm = data["valence"] > thr
    if existing_mask is not None:
        fm = np.logical_and(existing_mask, nm)
    return fm, None


def _make_neg_mask(data, *args, thr=0, existing_mask=None):
    nm = data["valence"] < thr
    if existing_mask is not None:
        fm = np.logical_and(existing_mask, nm)
    return fm, None


def _make_block_masks_pos(data, *args, **kwargs):
    train_mask, test_mask = _make_block_masks(data, *args, **kwargs)
    nm = data["valence"] > 0
    train_mask = np.logical_and(train_mask, nm)
    test_mask = np.logical_and(test_mask, nm)
    return train_mask, test_mask


def _make_block_masks_neg(data, *args, **kwargs):
    train_mask, test_mask = _make_block_masks(data, *args, **kwargs)
    nm = data["valence"] < 0
    train_mask = np.logical_and(train_mask, nm)
    test_mask = np.logical_and(test_mask, nm)
    return train_mask, test_mask


def _make_early_masks(
    data, trial_cutoff, trial_num_key="trial_num", existing_mask=None
):
    if existing_mask is None:
        existing_mask = np.ones(len(data), dtype=bool)
    train_mask = np.logical_and(existing_mask, data[trial_num_key] >= trial_cutoff)
    test_mask = np.logical_and(existing_mask, data[trial_num_key] < trial_cutoff)
    return train_mask, test_mask


def _binary_valence(valences):
    return valences > 0


def _identity_valence(valences):
    return valences


def _mean_split_valence(valences):
    return valences - np.mean(valences) > 0


def _integer_valence(valences):
    new_val = np.zeros_like(valences, dtype=int)
    u_vals = np.unique(valences)
    for i, uv in enumerate(u_vals):
        new_val[uv == valences] = i
    return new_val


def _mag_valence(valences):
    v_mags = np.abs(valences)
    return v_mags > np.nanmean(v_mags)


def _get_dim(pred_vid, targ, n_inds=20, pre_pca=.95, mean=True):
    preds = np.stack(list(pv[-n_inds:].flatten() for pv in pred_vid), axis=0)
    if pre_pca < 1:
        p = skd.PCA(pre_pca)
        preds = p.fit_transform(preds)
    m = skm.LinearSVC(dual="auto")
    m.fit(preds, targ)
    use_coeff = m.coef_
    if pre_pca < 1:
        use_coeff = p.inverse_transform(use_coeff)
    coeff_reshaped = np.reshape(use_coeff, (n_inds, pred_vid.iloc[0].shape[1]))
    if mean:
        coeff_reshaped = np.mean(coeff_reshaped, axis=0, keepdims=True)
    return coeff_reshaped, m.intercept_/n_inds


def make_dec_mask(data, used_pca, ind, mult, vid_shape=(240, 320), pred="video_1"):
    vid_shape = tuple(vid_shape)
    coeffs, intercepts, data = get_dec_dims(data, mean=False, pred=pred, ret_data=True)
    use_coeff = coeffs[ind]
    use_data, use_targ = data[ind]
    n_ts = coeffs.shape[1]
    n_trls = len(use_data)

    masks = np.zeros((n_trls, n_ts,) + vid_shape)
    for i in range(n_trls):
        vid_i = use_data.iloc[i][-n_ts:]
        weights = vid_i*use_coeff
        full_weights = used_pca.inverse_transform(weights)
        shaped_weights = np.reshape(full_weights, (n_ts,) + vid_shape)
        masks[i] = shaped_weights
    return masks, use_targ.to_numpy()


def make_dec_video(data, used_pca, ind, mult, vid_shape=(240, 320)):
    coeffs, intercepts = get_dec_dims(data, mean=False)
    use_coeff = coeffs[ind]
    n_ts = coeffs.shape[1]
    vid_pos = np.zeros((n_ts,) + tuple(vid_shape))
    vid_neg = np.zeros((n_ts,) + tuple(vid_shape))
    for i in range(n_ts):
        v_i_pos = used_pca.inverse_transform(mult*use_coeff[i])
        v_i_neg = used_pca.inverse_transform(-mult*use_coeff[i])
        vid_pos[i] = np.reshape(v_i_pos, vid_shape)
        vid_neg[i] = np.reshape(v_i_neg, vid_shape)
    return vid_pos, vid_neg


def get_dec_dims(data, pred='video_1', n_inds=20, ret_data=False, **kwargs):
    mask = ~pd.isna(data[pred])

    mask_val = mask
    targ_val = (data['valence'] > 0)[mask_val]
    data_val = data[mask_val][pred]
    val_dim, val_inter = _get_dim(
        data_val, targ_val, n_inds=n_inds, **kwargs,
    )

    mask_pos = np.logical_and(data['valence'] > 0, mask)
    targ_pos = (data['valence'] > .6)[mask_pos]
    data_pos = data[mask_pos][pred]
    pos_dim, pos_inter = _get_dim(
        data_pos, targ_pos, n_inds=n_inds, **kwargs,
    )

    mask_neg = np.logical_and(data['valence'] < 0, mask)
    targ_neg = (data['valence'] < -.6)[mask_neg]
    data_neg = data[mask_neg][pred]

    neg_dim, neg_inter = _get_dim(
        data_neg, targ_neg, n_inds=n_inds, **kwargs,
    )
    dims = np.stack((val_dim, pos_dim, neg_dim), axis=0)
    inters = np.stack((val_inter, pos_inter, neg_inter), axis=0)
    out = (dims, inters)
    if ret_data:
        pairs = ((data_val, targ_val), (data_pos, targ_pos), (data_neg, targ_neg))
        out = out + (pairs,)
    return out


def _flatten_X(X, last_vid_inds=None):
    col_groups = []
    for i in range(X.shape[1]):
        cg = X[:, i]
        if u.check_list(cg[0]):
            if len(cg[0].shape) > 1:
                vid_shape = cg[0].shape
                assert len(vid_shape) > 1
                if last_vid_inds is None:
                    lvi_i = np.min(
                        list(cg_i.shape[0] for cg_i in cg if cg_i is not None)
                    )
                else:
                    lvi_i = last_vid_inds

                new_vid = np.zeros((len(cg), lvi_i*vid_shape[1]))
                for j, cg_j in enumerate(cg):
                    if cg_j is None:
                        new_vid[j] = np.nan
                    else:
                        new_vid[j] = cg_j[-lvi_i:].flatten()
                col_groups.append(new_vid)
            else:
                col_groups.append(np.stack(cg, axis=0))
        else:
            cg = np.expand_dims(cg, 1)
            col_groups.append(cg)
    return np.concatenate(col_groups, axis=1).astype(float)


def decoding_param_sweep(param_name, param_vals, *args, n_cv=100, **kwargs):
    test_scores = np.zeros((len(param_vals), n_cv))
    gen_scores = np.zeros((len(param_vals), n_cv))

    for i, pv in enumerate(param_vals):
        kwargs[param_name] = pv
        out = decode_valence(*args, n_cv=n_cv, **kwargs)
        test_scores[i] = out['test_score']
        gen_scores[i] = out['test_mask_score']
    return test_scores, gen_scores


def decode_valence_tc(
    data,
    *args,
    predictors=default_predictors,
    valence_key="valence",
    shuffler=skms.ShuffleSplit,
    n_cv=100,
    test_frac=0.2,
    block_key="block",
    mask_func=_make_block_masks,
    model=skm.SVC,
    test_trls=None,
    keep_keys=None,
    target_func=_binary_valence,
    pre_pca=1,
    model_kwargs=None,
    compute_metrics=None,
    kernel="rbf",
    num_frames=None,
    trl_field="block_trial_num",
    **kwargs
):
    if model_kwargs is None:
        model_kwargs = {"kernel": kernel}
    if keep_keys is None:
        keep_keys = [
            valence_key,
        ]
    X = data[list(predictors)].to_numpy()
    X = _flatten_X(X, last_vid_inds=num_frames)
    y = target_func(data[valence_key].to_numpy())

    nan_mask = _make_nan_mask(X)
    train_mask, test_mask = mask_func(data, *args, existing_mask=nan_mask, **kwargs)

    pipe = na.make_model_pipeline(model=model, pca=pre_pca, **model_kwargs)

    cv = shuffler(n_cv, test_size=test_frac)
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    data_train = data[train_mask]
    data_test = data[test_mask][trl_field]

    fold_scores = {}
    fold_trls = {}
    test_scores = {}
    test_trls = {}
    for i, (tr_inds, te_inds) in enumerate(cv.split(X_tr, y_tr)):
        pipe.fit(X_tr[tr_inds], y_tr[tr_inds])
        score_split = pipe.predict(X_tr[te_inds]) == y_tr[te_inds]
        trls = data_train[trl_field].to_numpy()[te_inds]
        fold_scores[i] = score_split
        fold_trls[i] = trls

        score_test = pipe.predict(X_te) == y_te
        test_scores[i] = score_test
        test_trls[i] = data_test

    test_scores = np.stack(list(test_scores.values()))
    test_trls = np.stack(list(test_trls.values()))
    fold_scores = np.concatenate(list(fold_scores.values()))
    fold_trls = np.concatenate(list(fold_trls.values()))
    return (fold_scores, fold_trls), (np.mean(test_scores, axis=0), test_trls[0])


def decode_valence(
    data,
    *args,
    predictors=default_predictors,
    valence_key="valence",
    shuffler=skms.ShuffleSplit,
    n_cv=100,
    test_frac=0.2,
    block_key="block",
    mask_func=_make_block_masks,
    model=skm.SVC,
    test_trls=None,
    keep_keys=None,
    target_func=_binary_valence,
    pre_pca=1,
    model_kwargs=None,
    compute_metrics=None,
    kernel="rbf",
    num_frames=None,
    use_ica=False,
    **kwargs
):
    if model_kwargs is None:
        model_kwargs = {"kernel": kernel}
    if keep_keys is None:
        keep_keys = [
            valence_key,
        ]
    X = data[list(predictors)].to_numpy()
    X = _flatten_X(X, last_vid_inds=num_frames)
    y = target_func(data[valence_key].to_numpy())

    nan_mask = _make_nan_mask(X)
    train_mask, test_mask = mask_func(data, *args, existing_mask=nan_mask, **kwargs)

    pipe = na.make_model_pipeline(
        model=model, pca=pre_pca, use_ica=use_ica, **model_kwargs,
    )

    cv = shuffler(n_cv, test_size=test_frac)
    out = skms.cross_validate(
        pipe, X[train_mask], y[train_mask], cv=cv, return_estimator=True
    )

    if compute_metrics is not None:
        met_lists = {}
        for cm, met in compute_metrics.items():
            metrics_out = []
            for e in out["estimator"]:
                metrics_out.append(met(e, X[train_mask], y[train_mask]))
            met_lists[cm] = np.stack(metrics_out, axis=0)

        out["metric_scores"] = met_lists
    if test_mask is not None:
        out["test_mask_score"] = list(
            e.score(X[test_mask], y[test_mask]) for e in out["estimator"]
        )
        try:
            out["test_mask_proj"] = list(
                e.decision_function(X[test_mask]) for e in out["estimator"]
            )
        except AttributeError:
            out["test_mask_proj"] = list(e.predict(X[test_mask])
                                         for e in out["estimator"])

        out["test_mask_keys"] = data[keep_keys][test_mask]

    return out


def decode_valence_time(*args, trial_cutoff=50, **kwargs):
    return decode_valence(
        *args,
        trial_cutoff,
        mask_func=_make_early_masks,
        keep_keys=["valence", "trial_num"],
        **kwargs
    )


def decode_valence_block(*args, train_block=1, test_block=2, **kwargs):
    return decode_valence(
        *args,
        train_block,
        test_block,
        mask_func=_make_block_masks,
        keep_keys=["valence", "trial_num"],
        **kwargs
    )


def decode_valence_mag(*args, train_block=1, test_block=2, use_mags="pos",
                       use_blocks=True, **kwargs):
    if use_mags == "pos":
        if use_blocks:
            mask_func = _make_block_masks_pos
        else:
            mask_func = _make_pos_mask
    elif use_mags == "neg":
        if use_blocks:
            mask_func = _make_block_masks_neg
        else:
            mask_func = _make_neg_mask
    else:
        mask_func = _make_block_masks
    return decode_valence(
        *args,
        train_block,
        test_block,
        target_func=_mag_valence,
        mask_func=mask_func,
        keep_keys=["valence", "trial_num"],
        **kwargs
    )


default_dec_dict = {
    "block 1 to 2": (decode_valence_block, {"train_block": 1, "test_block": 2}),
    "block 2 to 1": (decode_valence_block, {"train_block": 2, "test_block": 1}),
    "time": (decode_valence_time, {"trial_cutoff": 50}),
    "mag_pos": (decode_valence_mag, {"train_block": 1, "test_block": 2,
                                     "use_mags": "pos", "use_blocks": False}),
    "mag_neg": (decode_valence_mag, {"train_block": 1, "test_block": 2,
                                     "use_mags": "neg", "use_blocks": False}),
}


def decode_valence_all(*args, dec_dict=default_dec_dict, **kwargs):
    out_dict = {}
    for k, (func, dec_kwargs) in dec_dict.items():
        full_kwargs = {}
        full_kwargs.update(dec_kwargs)
        full_kwargs.update(kwargs)
        out_dict[k] = func(*args, **full_kwargs)
    return out_dict
