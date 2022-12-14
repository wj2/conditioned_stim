
import argparse
import os
import pickle

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
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    data_file = args.data_file
    if args.video_folder is None:
        path, _ = os.path.split(data_file)
        video_folder = os.path.join(path, 'video_files')
    else:
        video_folder = args.video_folder

    vs = csx.process_videos(video_folder)
    data = csx.load_data(data_file, folder='')
    data = csx.preprocess_data(data, video_data=vs)

    pickle.dump(data, open(args.output_file, 'wb'))
