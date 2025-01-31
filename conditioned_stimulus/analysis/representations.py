import matplotlib.pyplot as plt
import general.plotting as gpl

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
    data_use, block_key="Block", block=None, valence_key="valence", mag_thr=.5,
):
    m1 = (data_use[valence_key] > mag_thr).rs_or(data_use[valence_key] < -mag_thr)
    m2 = m1.rs_not()
    if block is not None:
        block_mask = data_use[block_key] == block
        m1 = m1.rs_and(block_mask)
        m2 = m2.rs_and(block_mask)
    return m1, m2    


def make_valence_masks(
    data_use, block_key="Block", block=None, valence_key="valence", val_thr=.5,
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
            n_vars, n_times, figsize=(fwid * n_times, fwid * n_vars), sharey=True,
        )
    for i, (k_var, var_dict) in enumerate(out_dict.items()):
        for j, (time, out) in enumerate(var_dict.items()):
            dec, xs = out[:2]
            if line_labels is None:
                line_labels = ("",) * len(dec)
            for k, dec_k in enumerate(dec):
                gpl.plot_trace_werr(
                    xs, dec_k, confstd=True, ax=axs[i, j], label=line_labels[k],
                )
            gpl.add_hlines(.5, axs[i, j])
    return axs
