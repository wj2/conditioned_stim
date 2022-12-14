
import numpy as np
import sklearn.svm as skm
import sklearn.model_selection as skms

import general.neural_analysis as na 

default_predictors = (
    'sum_lick_count_window', 'sum_blink_count_window',
)
history_predictors = (
    'reward_1_back',
    'reward_2_back', 'reward_3_back', 'reward_4_back', 'reward_5_back',
    'airpuff_1_back', 'airpuff_2_back', 'airpuff_3_back', 'airpuff_4_back',
    'airpuff_5_back'
)

def _make_nan_mask(X):
    nan_mask = np.logical_not(np.any(np.isnan(X), axis=1))
    return nan_mask

def _make_block_masks(data, train_block, test_block, block_key='block',
                      existing_mask=None):
    if existing_mask is None:
        existing_mask = np.ones(len(data), dtype=bool)
    train_mask = np.logical_and(existing_mask, data[block_key] == train_block)
    test_mask = np.logical_and(existing_mask, data[block_key] == test_block)
    return train_mask, test_mask

def _make_block_masks_pos(data, *args, **kwargs):
    train_mask, test_mask = _make_block_masks(data, *args, **kwargs)
    nm = data['valence'] > 0
    train_mask = np.logical_and(train_mask, nm)
    test_mask = np.logical_and(test_mask, nm)
    return train_mask, test_mask

def _make_early_masks(data, trial_cutoff, trial_num_key='trial_num',
                      existing_mask=None):
    if existing_mask is None:
        existing_mask = np.ones(len(data), dtype=bool)
    train_mask = np.logical_and(existing_mask,
                                data[trial_num_key] >= trial_cutoff)
    test_mask = np.logical_and(existing_mask,
                               data[trial_num_key] < trial_cutoff)
    return train_mask, test_mask    

def _binary_valence(valences):
    return valences > 0

def _identity_valence(valences):
    return valences

def _mag_valence(valences):
    v_mags = np.abs(valences)
    return v_mags > np.nanmean(v_mags)

def _flatten_X(X):
    col_groups = []
    for i in range(X.shape[1]):
        cg = X[:, i]
        try:
            vid_shape = cg[0].shape
            assert len(vid_shape) > 0
            new_vid = np.zeros((len(cg), vid_shape[0]))
            for j, cg_j in enumerate(cg):
                new_vid[j] = cg_j
            col_groups.append(new_vid)
        except:
            cg = np.expand_dims(cg, 1)
            col_groups.append(cg)
    return np.concatenate(col_groups, axis=1).astype(float)

def decode_valence(data, *args, predictors=default_predictors,
                   valence_key='valence', shuffler=skms.ShuffleSplit,
                   n_cv=100, test_frac=.1, block_key='block',
                   mask_func=_make_block_masks,
                   model=skm.SVC, test_trls=None, keep_keys=None,
                   target_func=_binary_valence,
                   pre_pca=.99,
                   kernel='rbf', **kwargs):
    if keep_keys is None:
        keep_keys = [valence_key,]
    X = data[list(predictors)].to_numpy()
    X = _flatten_X(X)
    y = target_func(data[valence_key].to_numpy())

    nan_mask = _make_nan_mask(X)
    train_mask, test_mask = mask_func(data, *args, existing_mask=nan_mask,
                                      **kwargs)

    pipe = na.make_model_pipeline(model=model, kernel=kernel, pca=pre_pca)

    cv = shuffler(n_cv, test_size=test_frac)
    out = skms.cross_validate(pipe, X[train_mask], y[train_mask], cv=cv,
                              return_estimator=True)
    
    out['test_mask_score'] = list(e.score(X[test_mask], y[test_mask])
                                   for e in out['estimator'])
    try:
        out['test_mask_proj'] = list(e.decision_function(X[test_mask])
                                     for e in out['estimator'])
    except AttributeError:
        out['test_mask_proj'] = list(e.predict(X[test_mask])
                                     for e in out['estimator'])
        
    out['test_mask_keys'] = data[keep_keys][test_mask]

    return out

def decode_valence_time(*args, trial_cutoff=50, **kwargs):
    return decode_valence(*args, trial_cutoff, mask_func=_make_early_masks,
                          keep_keys=['valence', 'trial_num'], **kwargs)

def decode_valence_block(*args, train_block=1, test_block=2, **kwargs):
    return decode_valence(*args, train_block, test_block,
                          mask_func=_make_block_masks,
                          keep_keys=['valence', 'trial_num'], **kwargs)

def decode_valence_mag(*args, train_block=1, test_block=2, use_pos=True,
                       **kwargs):
    if use_pos:
        mask_func = _make_block_masks_pos
    else:
        mask_func = _make_block_masks
    return decode_valence(*args, train_block, test_block,
                          target_func=_mag_valence,
                          mask_func=mask_func,
                          keep_keys=['valence', 'trial_num'], **kwargs)

default_dec_dict = {
    'block 1 to 2':(decode_valence_block, {'train_block':1, 'test_block':2}),
    'block 2 to 1':(decode_valence_block, {'train_block':2, 'test_block':1}),
    'time': (decode_valence_time, {'trial_cutoff':50}),
    'mag': (decode_valence_mag, {'train_block':1, 'test_block':2}),
}
def decode_valence_all(*args, dec_dict=default_dec_dict, **kwargs):
    out_dict = {}
    for k, (func, dec_kwargs) in dec_dict.items():
        full_kwargs = {}
        full_kwargs.update(dec_kwargs)
        full_kwargs.update(kwargs)
        out_dict[k] = func(*args, **full_kwargs)
    return out_dict
