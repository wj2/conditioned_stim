
import argparse
import pickle
import numpy as np

import auxiliary as csx


def create_parser():
    parser = argparse.ArgumentParser(description='fit several autoencoders')
    parser.add_argument('video_folder', help='folder with videos')
    parser.add_argument('output_file', type=str,
                        help='path to save the output at')
    parser.add_argument('--max_load', default=np.inf, type=float,
                        help='maximum number of videos to load, for testing')
    parser.add_argument('--video_file_template', default=csx.video_template,
                        type=str, help='regex to use to find video files')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    vs = csx.process_videos(args.video_folder, max_load=args.max_load)

    pickle.dump(vs, open(args.output_file, 'wb'))
