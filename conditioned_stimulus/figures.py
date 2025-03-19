import numpy as np

import general.plotting as gpl
import general.paper_utilities as pu
import general.utility as u
import general.neural_analysis as na
import conditioned_stimulus.auxiliary as csx
import conditioned_stimulus.analysis.representations as csar


config_path = "conditioned_stimulus/conditioned_stimulus/figures.conf"


colors = (
    np.array(
        [
            (127, 205, 187),
            (65, 182, 196),
            (29, 145, 192),
            (34, 94, 168),
            (37, 52, 148),
            (8, 29, 88),
        ]
    )
    / 256
)


class CondStimFigure(pu.Figure):
    def _get_experimental_data(self, reinforcement_only=True, **kwargs):
        if self.exp_data is None:
            data = csx.load_sessions()
            if reinforcement_only:
                data = data.mask(data["reinforcement_trial"] == 1)
            self.exp_data = data
        return self.exp_data


class TimeGeneralizationFigure(CondStimFigure):
    def __init__(
        self, fig_key="time_generalization", exp_data=None, colors=colors, **kwargs
    ):
        fsize = (8, 21)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        self.exp_data = exp_data

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        tg_grid = pu.make_mxn_gridspec(self.gs, 6, 2, 0, 100, 0, 100, 10, 10)
        pkeys = (
            "panel_valence_time_gen",
            "panel_magnitude_time_gen",
            "panel_positive_magnitude_time_gen",
            "panel_negative_magnitude_time_gen",
            "panel_small_time_gen",
            "panel_large_time_gen",
        )
        axs = self.get_axs(tg_grid, squeeze=True)
        for i, axs_i in enumerate(axs):
            gss[pkeys[i]] = axs_i

        self.gss = gss

    def get_rep_bhv_info(self, key):
        if self.data.get((key, "info")) is None:
            t_start = self.params.getfloat("t_start")
            t_end = self.params.getfloat("t_end")
            binsize = self.params.getfloat("binsize")
            binstep = self.params.getfloat("binstep")
            tzf = self.params.get("tzf")
            data = self._get_experimental_data()

            info = csar.get_bhv_rep_dec_info(
                data,
                t_start,
                t_end=t_end,
                binsize=binsize,
                binstep=binstep,
                time_zero_field=tzf,
            )
            self.data[(key, "info")] = info
        return self.data[(key, "info")]

    def _time_generalization(self, key, sess_ind, target_mask_func, gen=False):
        axs = self.gss[key]
        (rep, xs_r), (bhv, xs_b), valence = self.get_rep_bhv_info(key)
        
        vmax = self.params.getfloat("vmax")
        vmin = self.params.getfloat("vmin")
        n_folds = self.params.getint("n_folds")
        if self.data.get((key, sess_ind, "dec")) is None:
            if gen:
                target, tr_mask, gen_mask = target_mask_func(valence[sess_ind])
                tr_rep = rep[sess_ind][tr_mask]
                tr_bhv = bhv[sess_ind][tr_mask]
                tr_targ = target[tr_mask]
                gen_rep = rep[sess_ind][gen_mask]
                gen_bhv = bhv[sess_ind][gen_mask]
                gen_targ = target[gen_mask]
                rep_kwargs = {"c_gen": gen_rep, "l_gen": gen_targ}
                bhv_kwargs = {"c_gen": gen_bhv, "l_gen": gen_targ}
            else:
                target, tr_mask = target_mask_func(valence[sess_ind])
                tr_rep = rep[sess_ind][tr_mask]
                tr_bhv = bhv[sess_ind][tr_mask]
                tr_targ = target[tr_mask]
                rep_kwargs = {}
                bhv_kwargs = {}
            out_rep = na.fold_skl_shape(
                tr_rep, tr_targ, n_folds, time_generalization=True, **rep_kwargs,
            )
            out_bhv = na.fold_skl_shape(
                tr_bhv, tr_targ, n_folds, time_generalization=True, **bhv_kwargs,
            )
            self.data[(key, sess_ind, "dec")] = (out_rep, out_bhv)
        out_rep, out_bhv = self.data[(key, sess_ind, "dec")]
        csar.plot_heat_maps(
            (out_rep, xs_r),
            (out_bhv, xs_b),
            faxs=(self.f, axs),
            titles=("neural", "behavioral"),
            vmax=vmax,
            vmin=vmin,
            plot_gen=gen,
        )

    def panel_valence_time_gen(self, sess_ind=0):
        key = "panel_valence_time_gen"

        def tm_func(val):
            mask = np.ones_like(val, dtype=bool)
            targ = val > 0
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func)

    def panel_magnitude_time_gen(self, sess_ind=0):
        key = "panel_magnitude_time_gen"

        def tm_func(val):
            mask = np.ones_like(val, dtype=bool)
            targ = np.abs(val) > 0.75
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func)

    def panel_positive_magnitude_time_gen(self, sess_ind=0):
        key = "panel_positive_magnitude_time_gen"

        def tm_func(val):
            mask = val > 0
            targ = np.abs(val) > 0.75
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func)

    def panel_negative_magnitude_time_gen(self, sess_ind=0):
        key = "panel_negative_magnitude_time_gen"

        def tm_func(val):
            mask = val < 0
            targ = np.abs(val) > 0.75
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func)

    def panel_small_gen_time_gen(self, sess_ind=0):
        key = "panel_small_time_gen"

        def tm_func(val):
            mask = np.abs(val) < .75
            targ = val > 0
            gen_mask = np.abs(val) > .75
            return targ, mask, gen_mask

        self._time_generalization(key, sess_ind, tm_func, gen=True)

    def panel_large_gen_time_gen(self, sess_ind=0):
        key = "panel_large_time_gen"

        def tm_func(val):
            mask = np.abs(val) > .75
            targ = val > 0
            gen_mask = np.abs(val) < .75
            return targ, mask, gen_mask

        self._time_generalization(key, sess_ind, tm_func, gen=True)

    def panel_small_time_gen(self, sess_ind=0):
        key = "panel_small_time_gen"

        def tm_func(val):
            mask = np.abs(val) < .75
            targ = val > 0
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func)

    def panel_large_time_gen(self, sess_ind=0):
        key = "panel_large_time_gen"

        def tm_func(val):
            mask = np.abs(val) > .75
            targ = val > 0
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func)
