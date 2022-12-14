
import numpy as np
import matplotlib.pyplot as plt

import general.plotting as gpl

def plot_corr_time(proj, true_val, trial_num, ax=None, plot_err=False, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    bin_val = np.where(true_val > 0, 1, -1)
    for proj_i in proj:
        corr_proj = proj_i*true_val
        if plot_err:
            corr_proj = corr_proj > 0
        gpl.plot_scatter_average(trial_num, corr_proj, ax=ax, **kwargs)
    return ax
