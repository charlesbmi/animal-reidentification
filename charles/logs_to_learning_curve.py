"""Read in AWS Sagemaker logs and plot the learning curves."""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import argparse
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description='Read in AWS Sagemaker logs.')
    parser.add_argument('-f', '--log-folder', type=pathlib.Path,
                        default='../logs',
                        help='Path to folder with csv logs.')

    return parser.parse_args()


def main():
    args = parse_args()

    log_files = args.log_folder.glob('*.csv')
    for log_file in log_files:
        df = pd.read_csv(log_file)
        train_df = df.loc[df.message.str.contains('train')]
        train_df = train_df.set_index(np.arange(1, len(train_df) + 1))

        val_df = df.loc[df.message.str.contains('val')]
        val_df = val_df.set_index(np.arange(1, len(val_df) + 1))

        train_df['avg_loss'] = train_df.message.apply(lambda x: float(x[33:39]))
        val_df['avg_loss'] = val_df.message.apply(lambda x: float(x[31:37]))

        train_df['accuracy'] = train_df.message.apply(lambda x: int(x[-5:-3]) / 100)
        val_df['accuracy'] = val_df.message.apply(lambda x: int(x[-5:-3]) / 100)
        train_df['error_rate'] = 1 - train_df['accuracy']
        val_df['error_rate'] = 1 - val_df['accuracy']

        fig, axs = plt.subplots(2, sharex=True, figsize=(10,10))
        fig.suptitle(log_file.stem)

        cols = ['avg_loss', 'error_rate']
        # Corresponding ylims
        ylims = [(9e-3, 3e0), (1.3e-1, 4.5e-1)]


        for col_idx, col in enumerate(cols):
            axs[col_idx].plot(train_df.index, train_df[col], label='train')
            axs[col_idx].plot(val_df.index, val_df[col], label='val')
            axs[col_idx].set_title(col)
            axs[col_idx].set_ylim(ylims[col_idx])
            axs[col_idx].legend()

        for ax in axs:
            ax.set_yscale('log')

        fig.savefig(log_file.stem + '.png')


if __name__ == '__main__':
    main()
