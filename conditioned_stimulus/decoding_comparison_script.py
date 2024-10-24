

import argparse
import os
import pickle

import matplotlib.pyplot as plt

import conditioned_stimulus.auxiliary as csx
import conditioned_stimulus.analysis as csa
import conditioned_stimulus.visualization as csv


def create_parser():
    parser = argparse.ArgumentParser(
        description='do decoding analysis for different features'
    )
    parser.add_argument('data_file', help='path to session_df to use')
    parser.add_argument('-o', '--output_name', default='decoding_analysis',
                        type=str,
                        help='path to save the output at (will be saved in data folder)')
    parser.add_argument("--pre_pca", default=None, type=float)
    parser.add_argument("--subplot_fwid", default=3, type=float)
    parser.add_argument("--fig_ext", default="svg")
    parser.add_argument("--n_cv", default=10, type=int)
    parser.add_argument("--save_decoding_data", default=False, action="store_true")
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    data = pickle.load(open(args.data_file, "rb"))

    if args.pre_pca >= 1:
        args.pre_pca = int(args.pre_pca)
        # add pre_pca value to output name
        args.output_name = "_".join((args.output_name, f"_{args.pre_pca}"))
    dec_data = csa.decode_feature_importance(data, pre_pca=args.pre_pca, n_cv=args.n_cv)
    
    fwid = args.subplot_fwid
    n_plots = len(list(dec_data.values())[0])
    f, axs = plt.subplots(1, n_plots, figsize=(n_plots*fwid, fwid), sharey="all")
    csv.plot_fi_dict(dec_data, axs=axs)

    path, _ = os.path.split(args.data_file)
    fig_path = os.path.join(path, ".".join((args.output_name, args.fig_ext)))
    f.savefig(fig_path, bbox_inches="tight", transparent=True)

    if args.save_decoding_data:
        dec_path = os.path.join(path, ".".join((args.output_name, "pkl")))
        pickle.dump(dec_data, open(dec_path, "wb"))
