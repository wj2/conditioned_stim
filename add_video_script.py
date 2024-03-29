
import argparse
import os
import pickle

import numpy as np

import conditioned_stimulus.auxiliary as csx


def create_parser():
    parser = argparse.ArgumentParser(description='fit several autoencoders')
    parser.add_argument('data_file', help='path to session_df to use')
    parser.add_argument('-o', '--output_file', default='session_df_video.pkl',
                        type=str,
                        help='path to save the output at')
    parser.add_argument('--video_folder', default=None,
                        help='path to video files -- will use "video_files" subfolder'
                        ' of the folder holding the data if not supplied')
    parser.add_argument("--no_intermediate_videos", default=True, action="store_false")
    parser.add_argument("--video_save_file", default=None)
    parser.add_argument("--max_load", default=np.inf, type=float)
    parser.add_argument("--ignore_saved", default=False, action="store_true")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    data_file = args.data_file
    video_folder = args.video_folder

    # video data
    print(f'Loading videos from {args.video_folder}')
    print(f'  number of videos: {len(os.listdir(args.video_folder))}')
    max_load = args.max_load
    if args.video_folder is None:
        path, _ = os.path.split(data_file)
        video_folder = os.path.join(path, 'video_files')
    else:
        video_folder = args.video_folder
    if args.video_save_file is None:
        folder = os.path.split(args.output_file)[0]
        vsf = os.path.join(folder, "processed-videos.pkl")
    if os.path.isfile(vsf) and not args.ignore_saved:
        vs, ms = pickle.load(open(vsf, "rb"))
    else:
        print(f'Loading data from {data_file}')
        try:
            vs = csx.process_videos(video_folder, max_load=max_load)
            print('  Done.')
        except Exception as e:
            print("loading videos failed, with {}".format(e))
            vs = None
        try:
            ms = csx.process_markers(video_folder, max_load=max_load)
        except Exception as e:
            print("loading markers failed, with {}".format(e))
            ms = None
        pickle.dump((vs, ms), open(vsf, "wb"))

    data = csx.load_data(data_file, folder='')
    print(f'  number of trials: {len(data)}')
    print(f'  Done.')

    # preprocess data
    print('Preprocessing data')
    data = csx.preprocess_data(data, video_data=vs[0], marker_data=ms)
    print('  Done.')
    print(f'Saving data to {args.output_file}')

    pickle.dump(data, open(args.output_file, 'wb'))
