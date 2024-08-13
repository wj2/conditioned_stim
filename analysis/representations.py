


def make_magnitude_masks(
    data_use, block_key="Block", block=None, valence_key="valence", mag_thr=.5,
):
    if block is not None:
        data_use = data_use.mask(data_use[block_key] == block)
    m1 = (data_use[valence_key] > mag_thr).rs_or(data_use[valence_key] < -mag_thr)
    m2 = m1.rs_not()
    return m1, m2    


def make_valence_masks(
    data_use, block_key="Block", block=None, valence_key="valence", val_thr=.5,
):
    if block is not None:
        data_use = data_use.mask(data_use[block_key] == block)
    m1 = m1 = data_use[valence_key] > val_thr
    m2 = m1.rs_not()
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
    
