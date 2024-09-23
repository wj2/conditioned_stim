import os
import time
import numpy as np
import skvideo.io as skv_io
import skvideo.utils as skv_u
import sklearn.decomposition as skd
import pandas as pd
import sklearn.utils as sku

import re

import general.data_io as gio
import general.utility as u

# PATHS
DATA_FOLD = "../data/conditioned_stim/"
FIG_FOLD = "conditioned_stimulus/figs/"
session_template = "(?P<animal>[a-zA-Z]+)_(?P<date>[0-9]+)"

conditioned_timing_rename_dict = {}
conditioned_info_rename_dict = {}


def load_session_files(
    folder,
    spikes="spike_times.pkl",
    bhv="[0-9]+_[a-z]+_(VR|airpuff)_behave\\.pkl",
    good_neurs="good_neurons.pkl",
    dlc_template="[0-9]+_[a-z]+_dlc_df_restruct.pkl",
):
    out_dict = {}
    out_dict["spikes"] = pd.read_pickle(open(os.path.join(folder, spikes), "rb"))
    bhv_fl = u.get_matching_files(folder, bhv)[0]
    out_dict["bhv"] = pd.read_pickle(open(bhv_fl, "rb"))
    try:
        dlc_fl = u.get_matching_files(folder, dlc_template)[0]
        dlc_df = pd.read_pickle(open(dlc_fl, "rb"))
        dlc_df["Trial"] = dlc_df["trial"]
        out_dict["dlc_markers"] = dlc_df
    except IndexError:
        print("no dlc file found in {}".format(folder))

    out_dict["good_neurs"] = pd.read_pickle(
        open(os.path.join(folder, good_neurs), "rb")
    )
    return out_dict


def organize_spikes(spikes, neur_info):
    neur_regions = tuple(neur_info["region"])
    n_trls = len(spikes)
    neur_regions_all = (neur_regions,) * n_trls
    spike_times = []
    for i, (_, row) in enumerate(spikes.iterrows()):
        spk_times_i = np.zeros(len(row), dtype=object)
        for j, r_ij in enumerate(row.to_numpy()):
            spk_times_i[j] = np.array(r_ij)
        spike_times.append(spk_times_i)
    return neur_regions_all, spike_times


def rename_fields(df, *dicts):
    full_dict = {}
    list(full_dict.update(d) for d in dicts)
    for old_name, new_name in full_dict.items():
        if u.check_list(old_name):
            present = list(on in df.columns for on in old_name)
            inds = np.where(present)[0]
            if len(inds) == 0:
                print("column {} is missing".format(old_name))
            else:
                old_name = old_name[inds[0]]
        if old_name in df.columns:
            rename = df[old_name]
        else:
            rename = np.ones(len(df)) * np.nan
            print("no {} in the columns".format(old_name))
        df[new_name] = rename
    return df


def mask_completed_trials(
    data,
    completed_field="completed_trial",
):
    mask = data[completed_field]
    return data.mask(mask)


def load_sessions(folder=DATA_FOLD, completed_only=True, **kwargs):
    data = gio.Dataset.from_readfunc(
        load_hashim_gulli_data_folder,
        folder,
        **kwargs,
    )
    if completed_only:
        data = mask_completed_trials(data)
    return data


default_make_numeric = (
    "Trace Start",
    "Trace End",
    "CS On",
)


def load_hashim_gulli_data_folder(
    folder,
    session_template=session_template,
    max_files=np.inf,
    exclude_last_n_trls=None,
    rename_dicts=None,
    load_only_nth_files=None,
    make_numeric=default_make_numeric,
):
    if rename_dicts is None:
        rename_dicts = (conditioned_timing_rename_dict, conditioned_info_rename_dict)
    dates = []
    monkeys = []
    n_neurs = []
    datas = []
    files_loaded = 0
    folder_gen = u.load_folder_regex_generator(
        folder,
        session_template,
        load_func=load_session_files,
        open_file=False,
        load_only_nth_files=load_only_nth_files,
    )
    for fl, fl_info, data_fl in folder_gen:
        dates.append(fl_info["date"])
        monkeys.append(fl_info["animal"])
        n_neurs.append(len(data_fl["good_neurs"]))
        neur_regions, spikes = organize_spikes(
            data_fl["spikes"],
            data_fl["good_neurs"],
        )
        data_all = data_fl["bhv"]["data_frame"]
        if "dlc_markers" in data_fl.keys():
            data_all = data_all.join(
                data_fl["dlc_markers"], on="Trial", rsuffix="dlc",
            )
        if len(data_all) > len(spikes):
            diff = len(data_all) - len(spikes)
            print(
                "difference in length between data ({}) and spikes ({})"
                "in file {}".format(len(data_all), len(spikes), fl)
            )
            data_all = data_all[:-diff].copy()
        data_all["spikeTimes"] = spikes
        data_all["neur_regions"] = neur_regions
        data_all["completed_trial"] = np.isin(data_all["TrialError"], (0, 6))
        data_all["correct_trial"] = data_all["TrialError"] == 0
        for field in make_numeric:
            data_all[field] = pd.to_numeric(data_all[field])
        data_all = rename_fields(data_all, *rename_dicts)
        datas.append(data_all)

        files_loaded += 1
        if files_loaded >= max_files:
            break
    super_dict = dict(
        date=dates,
        animal=monkeys,
        data=datas,
        n_neurs=n_neurs,
    )
    return super_dict


def load_data(fl, folder=DATA_FOLD, key="data_frame"):
    fp = os.path.join(folder, fl)
    #  No module named 'pandas.core.indexes.numeric'
    # return p.load(open(fp, "rb"))[key]
    return pd.read_pickle(fp)[key]


def extract_trial_number_default(file_name):
    parts = file_name.split("_")
    return (
        int(parts[2]) if len(parts) >= 3 else float("inf")
    )  # Use inf if no number is found


def _dim_red_video(video, use_pca=None, keep_pca=0.99):
    if use_pca is None:
        use_pca = skd.IncrementalPCA(keep_pca)
    use_pca.partial_fit(video)
    return use_pca, use_pca.transform(video)


# aragorn_230929_10_e3v831bDLC_resnet50_230929_aragorn_
# body_2Oct23shuffle1_1030000_filtered_labeled.mp4
video_name_template = (
    "(?P<monkey>[A-Za-z]+)_(?P<date>[0-9]+)_(?P<trial>[0-9]+)_(?P<cam>[a-z0-9]+)" ".*"
)
# video_name_template = (
#     "(?P<date>[0-9]+)_(?P<monkey>[A-Za-z]+)_(airpuff|choice)_Cam(?P<cam>[0-9]+)"
#     "_(?P<trial>[0-9]+).?"
# )
video_template = video_name_template + "\\.mp4"

# marker_name_template = (
#    "(?P<date>[0-9]+)_(?P<monkey>[A-Za-z]+)_(airpuff|choice)_Cam(?P<cam>[0-9]+)"
#      "_(?P<trial>[0-9]+).*_filtered"
# )
marker_name_template = (
    "(?P<monkey>[A-Za-z]+)_(?P<date>[0-9]+)_(?P<trial>[0-9]+)_(?P<cam>[a-z0-9]+)"
    ".*_filtered"
)
marker_template = marker_name_template + "\\.csv"


def _interpret_file(
    fl, groups=("date", "trial", "cam", "monkey"), file_template=marker_template
):
    m = re.match(file_template, fl)
    if m is not None:
        out = tuple(m.group(g) for g in groups)
    else:
        out = None
    return out


def interpret_video_file(fl, file_template=video_template):
    return _interpret_file(
        fl,
        file_template=file_template,
        groups=("date", "trial", "cam", "monkey"),
    )


def interpret_marker_file(*args, **kwargs):
    return _interpret_file(*args, **kwargs)


def _marker_generator(
    folder,
    file_template=marker_template,
    max_load=np.inf,
    data=None,
    trial_key="Trial",
    header_lines=(0, 1, 2),
    start_key=None,
    end_key=None,
    frame_key="cam_frames",
    trial_start_key="Start Trial",
):
    fls = os.listdir(folder)
    loaded = 0
    for fl in fls:
        out = interpret_marker_file(fl, file_template=file_template)
        if out is not None:
            date, trial, cam, monkey = out
            print(f"  marker {fl}")
            markers = pd.read_csv(os.path.join(folder, fl), header=list(header_lines))
            # if data is not None and start_key is not None and end_key is not None:
            #     mask = data[trial_key].to_numpy() == int(trial) + 1
            #     trl_data = data[mask]
            #     start = trl_data[start_key].iloc[0]
            #     end = trl_data[end_key].iloc[0]

            #     if pd.isna(start) or pd.isna(end):
            #         print(f"  skipping {fl} because {start} or {end} is nan")
            #         continue
            #     frame_times = np.array(trl_data[frame_key].iloc[0])
            #     trl_start = trl_data[trial_start_key].to_numpy()
            #     frame_times = frame_times[frame_times > trl_start]
            loaded += 1
            yield out, markers
        if loaded >= max_load:
            break


def _video_generator(
    folder,
    file_template=video_template,
    max_load=np.inf,
    reduce=2,
    data=None,
    trial_key="Trial",
    start_key=None,
    end_key=None,
    frame_key="cam_frames",
    trial_start_key="Start Trial",
):
    # sort files by trial number
    fls = sorted(os.listdir(folder), key=extract_trial_number_default)

    loaded = 0
    for fl in fls:
        if "filtered" in fl:
            print(f"     skipping {fl}")
            continue
        out = interpret_video_file(fl, file_template)
        if out is not None:
            date, trial, cam, monkey = out
            print(f"trial {trial}, monkey {monkey}, cam {cam}, ")
            start_time = time.time()
            video = skv_io.vread(os.path.join(folder, fl))
            video = skv_u.rgb2gray(video)
            video = video[:, ::reduce, ::reduce]
            print(video.shape)
            if data is not None and start_key is not None and end_key is not None:
                mask = data[trial_key].to_numpy() == int(trial) + 1
                trl_data = data[mask]
                start = trl_data[start_key].iloc[0]
                end = trl_data[end_key].iloc[0]

                if pd.isna(start) or pd.isna(end):
                    print(f"  skipping {fl} because {start} or {end} is nan")
                    continue
                frame_times = np.array(trl_data[frame_key].iloc[0])
                trl_start = trl_data[trial_start_key].to_numpy()
                frame_times = frame_times[frame_times > trl_start]
                ft_len = frame_times.shape[0]
                vid_len = video.shape[0]
                if ft_len != vid_len:
                    print(
                        "  frame times and video are different lengths: {}, {}".format(
                            ft_len, vid_len
                        )
                    )
                frame_times = frame_times[: video.shape[0]]
                video_mask = np.logical_and(start <= frame_times, end > frame_times)
                video = video[video_mask]
            try:
                video = np.reshape(video, (video.shape[0], -1))
                end_time = time.time()
                print(f"     video {fl} loaded in {round(end_time - start_time, 2)}s")
                loaded += 1
                yield (date, trial, cam, monkey), video
            except ValueError as e:
                print(f"     error reshaping {fl}: {e}")
                continue
        if loaded >= max_load:
            break


def process_markers(
    folder,
    file_template=marker_template,
    max_load=np.inf,
    data=None,
    epoch_start=None,
    epoch_end=None,
):
    print(f"Epochs Selected: {epoch_start}-{epoch_end}")
    markers_all = {}
    for info, markers in _marker_generator(
        folder,
        file_template=file_template,
        max_load=max_load,
        data=data,
        start_key=epoch_start,
        end_key=epoch_end,
        frame_key="cam_frames",
        trial_start_key="Start Trial",
    ):
        date, trial, cam, monkey = info
        curr = markers_all.get((date, trial, monkey), {})
        curr[cam] = markers.to_numpy()
        markers_all[(date, monkey, trial)] = curr
    print(f"  Markers found: {len(markers_all)}")
    return markers_all


def _batch_partial_fit(pca, video, batch_size=100):
    for batch in sku.gen_batches(video.shape[0], batch_size):
        pca.partial_fit(video[batch])
    return pca


def process_videos(
    folder,
    file_template=video_template,
    cams=("0", "1"),
    keep_pca=None,
    max_load=np.inf,
    data=None,
    epoch_start=None,
    epoch_end=None,
):
    """
    Reduce the dimensionality of the videos, while keeping as much variance as
    possible.

    The output of this function should be saved as a pickle file for use when
    loading the corresponding behavioral data file.

    Parameters
    ----------
    folder : str
        The relative or absolute path to the folder with video files.
    file_template : str, optional
        A regular expression that can be used to find the desired video files. It
        should also have the following named groups: "date", "monkey", "cam", "trial"
    cams : sequence of str, optional
        The expected cams.
    max_load : numeric, optional
        The function will stop loading videos once this is exceeded, just useful for
        testing.

    Returns
    -------
    videos : dictionary
        Dictionary with keys (date, monkey, trial) with values that are also
        dictionaries and have keys corresponding to different cameras with video
        values
    cam_pca : dictionary
        Dictionary with keys corresponding to cameras and values coresponding to
        the IncrementalPCA object for that camera.
    """
    videos = {}
    cam_pca = {}
    # time each for loop
    print(f"Epochs Selected: {epoch_start}-{epoch_end}")
    for info, video in _video_generator(
        folder,
        file_template=file_template,
        max_load=max_load,
        data=data,
        start_key=epoch_start,
        end_key=epoch_end,
    ):
        time_start = time.time()
        date, trial, cam, monkey = info
        use_pca = cam_pca.get(cam, skd.IncrementalPCA(keep_pca))
        use_pca = _batch_partial_fit(use_pca, video)
        cam_pca[cam] = use_pca
        # get time for each loop
        time_end = time.time()
        print(f"    pca time: {round(time_end - time_start, 2)}s")
    print("-----")
    for info, video in _video_generator(
        folder,
        file_template=file_template,
        max_load=max_load,
        data=data,
        start_key=epoch_start,
        end_key=epoch_end,
    ):
        date, trial, cam, monkey = info
        use_pca = cam_pca.get(cam)
        video_dr = use_pca.transform(video)

        trl_dict = videos.get((date, monkey, trial), {})
        trl_dict[cam] = video_dr
        videos[(date, monkey, trial)] = trl_dict
    return videos, cam_pca


def _ident_func(x, **kwargs):
    return x


def _add_video_data(
    data,
    video_data,
    video_key="video_{}",
    red_func=_ident_func,
    window_start="Trace Start",
    window_end="Delay Off",
    video_times_key="cam{}_trial_time",
    video_file_key="cam1_trial_name",
    vn_template=video_name_template,
    marker_data=None,
    marker_key="markers_{}",
    trial_key="Trial",
):
    print("Adding video data...")
    print(video_data.keys())
    cams = np.concatenate(list(list(vdv.keys()) for vdv in video_data.values()))
    cams = np.unique(cams)
    new_dict = {cam: [] for cam in cams}
    new_marker_dict = {cam: [] for cam in cams}
    if marker_data is None:
        marker_data = {}
    for _, row in data.iterrows():
        monkey = row["subject"]
        # videos are saved with index 0 but trial_num is indexed at 1
        trial = row[trial_key] - 1
        date = row["date"]
        # out = interpret_video_file(row[video_file_key], vn_template)
        # date, trial, _, monkey = out
        print(date, monkey, str(trial))
        vid_entry = video_data.get((date, monkey, str(trial)))
        markers_trl = marker_data.get((date, monkey, str(trial)), {})

        row_bounds = (row[window_start], row[window_end])
        print("  Trial {}".format(trial))
        if vid_entry is not None and not (
            pd.isnull(row_bounds[0]) or pd.isnull(row_bounds[1])
        ):
            for cam in cams:
                vid = vid_entry.get(cam, None)
                markers = markers_trl.get(cam)
                vid_list = new_dict.get(cam, [])
                marker_list = new_marker_dict.get(cam, [])
                vid_list.append(vid)
                if markers is not None:
                    marker_list.append(markers)
                    print("    cam {} markers added".format(cam))
                else:
                    marker_list.append(None)
                    print("    cam {} markers missing".format(cam))
                # video_times = row[video_times_key.format(cam)]
                # if video_times.shape[0] == vid.shape[0]:
                #     mask = np.logical_and(video_times >= row_bounds[0],
                #                           video_times < row_bounds[1])
                #     vid_list.append(vid[mask])
                #     if markers is not None:
                #         marker_list.append(markers[mask])
                #     else:
                #         marker_list.append(None)

                # else:
                #     print(video_times.shape[0], vid.shape[0])
                #     print('mismatched length', trial)
                #     vid_list.append(None)
                #     marker_list.append(None)
                new_dict[cam] = vid_list
                new_marker_dict[cam] = marker_list
                print("    cam {} video data added".format(cam))
        else:
            if vid_entry is None:
                print("missing vid", trial)
            else:
                print("null bounds", trial)
            for cam in cams:
                vid_list = new_dict.get(cam, [])
                vid_list.append(None)
                new_dict[cam] = vid_list

                marker_list = new_marker_dict.get(cam, [])
                marker_list.append(None)
                new_marker_dict[cam] = marker_list
    for k, d in new_dict.items():
        print("  data frame has {} length".format(len(data)))
        print(
            "  cam {} has {} vids and {} markers".format(
                k, len(d), len(new_marker_dict[k])
            )
        )
        if len(d) != len(data):
            print("    session and videos mismatched lengths")
            new_dict_keys = list(new_dict.keys())
            for k in new_dict_keys:
                print("    {}: {}".format(k, len(new_dict[k])))
        len_d = len(list(x for x in d if x is not None))
        print(f"  {len_d}")
        data[video_key.format(k)] = d
        data[marker_key.format(k)] = new_marker_dict[k]
    return data


def _mask_eyes(
    data,
    trace_field,
    map_field,
    ex="eye_x",
    ey="eye_y",
    start="Trace Start",
    end="Delay Off",
    eye_range=(-30, 30),
):
    masked_eyes = []
    eye_maps = []
    trls = data[[ex, ey, start, end]]
    for trl in trls.itertuples(index=False):
        ex_t, ey_t, start_t, end_t = trl
        eye = np.stack((ex_t, ey_t), axis=1)
        ts = np.arange(eye.shape[0])
        if pd.isna(start_t) or pd.isna(end_t):
            mask = np.ones_like(ts, dtype=bool)
        else:
            mask = np.logical_and(ts > start_t, ts < end_t)
        masked_eyes.append(eye[mask])
        eye_map = np.histogramdd(
            eye[mask],
            range=(eye_range,) * 2,
            density=True,
        )
        eye_maps.append(eye_map[0].flatten())
    data[trace_field] = masked_eyes
    data[map_field] = eye_maps
    return data


# add trial num within block
def preprocess_data(
    data,
    sum_fields=("lick_count_window", "blink_count_window"),
    session_marker="date",
    block_field="Block",
    tnum_field="Trial",
    reward_trigger="Reward Trigger",
    air_trigger="Airpuff Trigger",
    trace_trigger="Trace End",
    delay_off_field="Delay Off",
    eye_mask_field="eye_masked_xy",
    eye_map_field="eye_map_xy",
    video_data=None,
    marker_data=None,
    sum_windows=None,
):
    for sf in sum_fields:
        try:
            new_field = "sum_" + sf
            data[new_field] = list(np.sum(sf_i) for sf_i in data[sf])
        except Exception as e:
            print(f"  {sf} not found")
            print(e)
    data["positive_valence"] = data["reward"] > 0
    delay_off = np.zeros(len(data))
    delay_off[:] = np.nan
    tr_mask = ~pd.isna(data[trace_trigger])
    delay_off[tr_mask] = data[trace_trigger][tr_mask]
    rew_mask = ~pd.isna(data[reward_trigger])
    delay_off[rew_mask] = data[reward_trigger][rew_mask]
    air_mask = ~pd.isna(data[air_trigger])
    delay_off[air_mask] = data[air_trigger][air_mask]
    data[delay_off_field] = delay_off
    block_tnum = np.copy(data[tnum_field])
    for u_sess in np.unique(data[session_marker]):
        s_mask = data[session_marker] == u_sess
        for block in np.unique(data[block_field]):
            b_mask = data[block_field] == block
            mask = np.logical_and(s_mask, b_mask)
            sub = np.min(data[tnum_field][mask]) - 1
            block_tnum[mask] = block_tnum[mask] - sub
    data["block_trial_num"] = block_tnum
    # remap eye data fields
    if "eye_x" not in data.columns:
        data["eye_x"] = data["AnalogData.Eye.0"]
        data["eye_y"] = data["AnalogData.Eye.1"]
    data = _mask_eyes(data, eye_mask_field, eye_map_field)

    chose_side = np.ones(len(data))
    choice_mask = data["choice_trial"] == 1
    choice_mask = np.logical_and(choice_mask, data["fractal_chosen"] != "_error")
    chose_2 = data["fractal_chosen"] == data["stimuli_name_2"]
    chose_side += chose_2
    chose_side[~choice_mask] = 0
    data["chose_side"] = chose_side
    if "trial_num" not in data.columns:
        data["trial_num"] = data["Trial"].astype(int)
    if video_data is not None:
        data = _add_video_data(
            data,
            video_data,
            window_start="Start Trial",
            window_end="End Trial",
            marker_data=marker_data,
        )
    # for index, row in data[["pupil_data_window", "pupil_pre_CS"]].iterrows():
    #     (pw, pre) = row
    #     if np.all(np.isnan(pw)):
    #         pws.append(np.nan)
    #         pdiffs.append(np.nan)
    #     else:
    #         pw_i = pw[:]
    #         pw_i[pw_i == 0] = np.nan
    #         pws.append(pw_i)
    #         pdiffs.append(np.nanmean(pw_i[-500:]) - np.nanmean(pre))
    # data["pupil_diff"] = pdiffs
    # data["pupil_window_nan"] = pws
    return data


def summary_preprocessing(data):
    print("Summary of data preprocessing")
    print(f"  Total number of trials: {len(data)}")
    correct_trials = data[data["TrialError"] == 0]
    print(f"    Correct Trials: {len(correct_trials)}")
    # find all columns with 'video' in the name
    video_cols = [col for col in data.columns if "video" in col]
    print(f"  Number of cameras: {len(video_cols)}")
    for cam in video_cols:
        # count the number of non-null entries in each column
        num_vids = data[cam].apply(lambda x: x is not None).sum()
        print(f"    {cam} Data: {num_vids}")
    marker_cols = [col for col in data.columns if "markers" in col]
    print(f"  Number of cameras: {len(marker_cols)}")
    for cam in marker_cols:
        num_vids = data[cam].apply(lambda x: x is not None).sum()
        print(f"    {cam} Data: {num_vids}")
    print("  Done.")
    return data
