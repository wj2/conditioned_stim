
import os
import pickle as p
import numpy as np
import skvideo.io as skv_io
import skvideo.utils as skv_u
import sklearn.decomposition as skd

import re

## PATHS
DATA_FOLD = '../data/conditioned_stim/'
FIG_FOLD = 'conditioned_stimulus/figs/'

def load_data(fl, folder=DATA_FOLD):
    fp = os.path.join(folder, fl)
    return p.load(open(fp, 'rb'))

def _dim_red_video(video, use_pca=None, keep_pca=.99):
    if use_pca is None:
        use_pca = skd.PCA(keep_pca)
        use_pca.fit(video)
    return use_pca, use_pca.transform(video)    

video_template = ('(?P<date>[0-9]+)_(?P<monkey>[A-Za-z]+)_trial_(?P<trial>[0-9]+)'
                  '_cam(?P<cam>[0-9]+).mp4')
def process_videos(
        folder,
        file_template=video_template,
        cams=('0', '1'),
        keep_pca=1,
        combine_videos=False,
):
    fls = os.listdir(folder)
    videos = {}
    cam_pca = {}
    comb_pca = None
    full_vids = {}
    for fl in fls:
        m = re.match(file_template, fl)
        if m is not None:
            date = m.group('date')
            trial = m.group('trial')
            cam = m.group('cam')
            monkey = m.group('monkey')
            video = skv_io.vread(os.path.join(folder, fl))
            video = skv_u.rgb2gray(video)
            video = np.reshape(video, (video.shape[0], -1))
            cam_pca[cam], video = _dim_red_video(video, use_pca=cam_pca.get(cam),
                                                 keep_pca=keep_pca)
            trl_dict = videos.get((date, monkey, trial), {})
            trl_dict[cam] = video
            videos[(date, monkey, trial)] = trl_dict
            if len(trl_dict.keys()) == len(cams):
                if combine_videos:
                    full_vid = np.concatenate(list(trl_dict[cam] for cam in cams),
                                              axis=1)
                    add_obj = _dim_red_video(full_vid, use_pca=comb_pca,
                                             keep_pca=keep_pca)
                else:
                    add_obj = trl_dict
                d_dict = full_vids.get((date, monkey), {})
                d_dict[trial] = add_obj
                full_vids[(date, monkey)] = d_dict

                _ = videos.pop((date, monkey, trial))
    
    return full_vids

def _add_video_data(data, video_data, video_key='video_{}', red_func=np.sum):
    cams = list(video_data.values())[0]
    new_dict = {}
    for _, row in data.iterrows():
        date = row['date']
        monkey = row['subject']
        trial = row['trial_num'] - 1
        vids = video_data[(date, monkey)].get(str(trial))
        if vids is not None:
            for cam, vid in vids.items():
                vk = video_key.format(cam)
                vo = new_dict.get(vk, [])
                vid = red_func(vid, axis=0)
                vo.append(vid)
                new_dict[vk] = vo
        else:
            print('no {}, {}, {}'.format(date, monkey, trial))
            list(new_dict[k].append(np.nan) for k in new_dict.keys())
    for k, d in new_dict.items():
        data[k] = d
    return data
    
# add trial num within block
def preprocess_data(
        data,
        sum_fields=('lick_count_window', 'blink_count_window'),
        session_marker='date',
        block_field='block',
        tnum_field='trial_num',
        video_data=None,
):
    for sf in sum_fields:
        new_field = 'sum_' + sf
        data[new_field] = list(np.sum(sf_i) for sf_i in data[sf])
    data['positive_valence'] = data['reward_prob'] > 0
    pws = []
    pdiffs = []
    block_tnum = data[tnum_field][:]
    for u_sess in np.unique(data[session_marker]):
        s_mask = data[session_marker] == u_sess
        for block in np.unique(data[block_field]):
            b_mask = data[block_field] == block
            mask = np.logical_and(s_mask, b_mask)
            sub = np.min(data[tnum_field][mask]) - 1
            block_tnum[mask] = block_tnum[mask] - sub
    data['block_trial_num'] = block_tnum
    if video_data is not None:
        data = _add_video_data(data, video_data)
    for index, row in data[['pupil_window', 'pupil_pre_CS']].iterrows():
        (pw, pre) = row
        if np.all(np.isnan(pw)):
            pws.append(np.nan)
            pdiffs.append(np.nan)
        else:
            pw_i = pw[:]
            pw_i[pw_i == 0] = np.nan
            pws.append(pw_i)
            pdiffs.append(np.nanmean(pw_i[-500:]) - np.nanmean(pre))
    data['pupil_diff'] = pdiffs
    data['pupil_window_nan'] = pws
    return data
