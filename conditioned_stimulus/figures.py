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

    def get_matched_bhv_neural_data(
        self,
        t_start,
        t_end=None,
        binsize=500,
        binstep=20,
        regions=None,
        info_keys=None,
        tzf="CS On",
    ):
        data = self._get_experimental_data()
        (rep, xs_r), (bhv, xs_b), valence = csar.get_bhv_rep_dec_info(
            data,
            t_start,
            t_end=t_end,
            binsize=binsize,
            binstep=binstep,
            time_zero_field=tzf,
            regions=regions,
        )
        if np.any((xs_r - xs_b) ** 2 > 1e-2):
            raise IOError(
                "rep xs and bhv xs are not all close\nrep: {}\nbhv: {}".format(
                    xs_r, xs_b
                )
            )
        out = []
        for i, rep_i in enumerate(rep):
            out_i = {"neural": rep_i, "bhv": bhv[i], "xs": xs_r}
            if info_keys is not None:
                for ik in info_keys:
                    out_i[ik] = data[ik][i].to_numpy()
            out.append(out_i)
        return out


class DecodingFigure(CondStimFigure):
    def __init__(
        self, fig_key="decoding", regions=None, exp_data=None, colors=colors, **kwargs
    ):
        fsize = (21, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        self.exp_data = exp_data

        params = cf[fig_key]
        self.fig_key = fig_key
        self.regions = regions
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        tg_grid = pu.make_mxn_gridspec(self.gs, 2, 8, 0, 100, 0, 100, 5, 2)
        dec_axs = self.get_axs(tg_grid, squeeze=False, sharey="all")
        gss["panel_early_decoding"] = dec_axs[:, ::2]
        gss["panel_late_decoding"] = dec_axs[:, 1::2]

        self.gss = gss

    def _decoding(self, t_start, t_end, tzf, axs, key, recompute=False, late=False):
        binsize = self.params.getfloat("binsize")
        binstep = self.params.getfloat("binstep")
        n_folds = self.params.getint("n_folds")
        test_prop = self.params.getfloat("test_prop")
        train_values = self.params.getlist("train_values", typefunc=int)
        test_values = self.params.getlist("test_values", typefunc=int)
        gen_key = self.params.get("gen_key")

        dec_keys = (
            "valence_binary",
            "magnitude_binary",
            "fractal_binary1",
            "past_valence_binary",
        )
        data_types = ("neural", "bhv")

        info_keys = dec_keys + (gen_key,)

        if self.data.get(key) is None or recompute:
            outs = self.get_matched_bhv_neural_data(
                t_start,
                t_end,
                binsize=binsize,
                binstep=binstep,
                regions=self.regions,
                info_keys=info_keys,
                tzf=tzf,
            )
            dec_results = []
            for i, out in enumerate(outs):
                k_i = {}
                block = out[gen_key]
                if test_values is None and train_values is not None:
                    test_values_i = np.array(
                        set(np.unique(block)).difference(train_values)
                    )
                else:
                    test_values_i = test_values
                for j, dk in enumerate(dec_keys):
                    k_i[dk] = {}
                    for k, dt in enumerate(data_types):
                        if train_values is None:
                            X_tr = out[dt]
                            X_te = None
                            y_tr = out[dk]
                            y_te = None
                        else:
                            tr_mask = np.isin(block, train_values)
                            te_mask = np.isin(block, test_values_i)
                            X_tr = out[dt][tr_mask]
                            y_tr = out[dk][tr_mask]
                            X_te = out[dt][te_mask]
                            y_te = out[dk][te_mask]
                        dec_ijk = na.fold_skl_shape(
                            X_tr,
                            y_tr,
                            n_folds,
                            mean=False,
                            impute_missing=True,
                            c_gen=X_te,
                            l_gen=y_te,
                            test_prop=test_prop,
                        )
                        k_i[dk][dt] = (dec_ijk, out["xs"])
                dec_results.append(k_i)
            self.data[key] = dec_results

        dec_results = self.data[key]
        for i, res in enumerate(dec_results):
            for j, dk in enumerate(dec_keys):
                for k, dt in enumerate(data_types):
                    res_ijk, xs = res[dk][dt]
                    l_ = gpl.plot_trace_werr(
                        xs, res_ijk["score"], confstd=True, ax=axs[k, j]
                    )
                    if "score_gen" in res_ijk.keys():
                        gpl.plot_trace_werr(
                            xs,
                            res_ijk["score_gen"],
                            confstd=True,
                            ax=axs[k, j],
                            color=l_[0].get_color(),
                            plot_outline=True,
                            ls="dashed",
                        )
                    gpl.add_hlines(0.5, axs[k, j])
                    if k == 0 and not late:
                        axs[k, j].set_title("{} decoding".format(dk))
                    if j == 0 and not late:
                        axs[k, j].set_ylabel("decoding performance\n({})".format(dt))
                    if late:
                        gpl.clean_plot(axs[k, j], 1)
                    if k == len(data_types) - 1:
                        axs[k, j].set_xlabel("time from {}".format(tzf))

    def panel_early_decoding(self, recompute=False):
        key = "panel_early_decoding"
        axs = self.gss[key]

        t_start = self.params.getfloat("t_start_early")
        t_end = self.params.getfloat("t_end_early")
        tzf = self.params.get("early_tzf")

        return self._decoding(t_start, t_end, tzf, axs, key, recompute=recompute)

    def panel_late_decoding(self, recompute=False):
        key = "panel_late_decoding"
        axs = self.gss[key]

        t_start = self.params.getfloat("t_start_late")
        t_end = self.params.getfloat("t_end_late")
        tzf = self.params.get("late_tzf")

        return self._decoding(
            t_start, t_end, tzf, axs, key, recompute=recompute, late=True
        )


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

    def get_rep_bhv_info(self, key, **kwargs):
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
                **kwargs,
            )
            self.data[(key, "info")] = info
        return self.data[(key, "info")]

    def _time_generalization(
        self, key, sess_ind, target_mask_func, gen=False, **kwargs
    ):
        axs = self.gss[key]
        (rep, xs_r), (bhv, xs_b), valence = self.get_rep_bhv_info(key, **kwargs)

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
                tr_rep,
                tr_targ,
                n_folds,
                time_generalization=True,
                **rep_kwargs,
            )
            out_bhv = na.fold_skl_shape(
                tr_bhv,
                tr_targ,
                n_folds,
                time_generalization=True,
                **bhv_kwargs,
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

    def panel_valence_time_gen(self, sess_ind=0, **kwargs):
        key = "panel_valence_time_gen"

        def tm_func(val):
            mask = np.ones_like(val, dtype=bool)
            targ = val > 0
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func, **kwargs)

    def panel_magnitude_time_gen(self, sess_ind=0, **kwargs):
        key = "panel_magnitude_time_gen"

        def tm_func(val):
            mask = np.ones_like(val, dtype=bool)
            targ = np.abs(val) > 0.75
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func, **kwargs)

    def panel_positive_magnitude_time_gen(self, sess_ind=0, **kwargs):
        key = "panel_positive_magnitude_time_gen"

        def tm_func(val):
            mask = val > 0
            targ = np.abs(val) > 0.75
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func, **kwargs)

    def panel_negative_magnitude_time_gen(self, sess_ind=0, **kwargs):
        key = "panel_negative_magnitude_time_gen"

        def tm_func(val):
            mask = val < 0
            targ = np.abs(val) > 0.75
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func, **kwargs)

    def panel_small_gen_time_gen(self, sess_ind=0, **kwargs):
        key = "panel_small_time_gen"

        def tm_func(val):
            mask = np.abs(val) < 0.75
            targ = val > 0
            gen_mask = np.abs(val) > 0.75
            return targ, mask, gen_mask

        self._time_generalization(key, sess_ind, tm_func, gen=True, **kwargs)

    def panel_large_gen_time_gen(self, sess_ind=0, **kwargs):
        key = "panel_large_time_gen"

        def tm_func(val):
            mask = np.abs(val) > 0.75
            targ = val > 0
            gen_mask = np.abs(val) < 0.75
            return targ, mask, gen_mask

        self._time_generalization(key, sess_ind, tm_func, gen=True, **kwargs)

    def panel_small_time_gen(self, sess_ind=0, **kwargs):
        key = "panel_small_time_gen"

        def tm_func(val):
            mask = np.abs(val) < 0.75
            targ = val > 0
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func, **kwargs)

    def panel_large_time_gen(self, sess_ind=0, **kwargs):
        key = "panel_large_time_gen"

        def tm_func(val):
            mask = np.abs(val) > 0.75
            targ = val > 0
            return targ, mask

        self._time_generalization(key, sess_ind, tm_func, **kwargs)
