import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plot_custom_scale

sns.set()
sns.set_style('whitegrid')
sns.set_context('paper')


def perform_size_reduction_plot(df, x_lab):
    fig, ax = plt.subplots()
    plt.rcParams['font.family'] = 'Osaka'

    if x_lab == 'n':
        x_mod = 10
    else:
        x_mod = 0
    colors = ['dark royal blue', 'milk chocolate', 'dull blue', 'carmine', 'deep lavender']
    cmap = sns.color_palette(sns.xkcd_palette(colors))
    for i, (cls, mode) in enumerate([(0, 'VC'), (1, 'PC'), (3, 'PS'), (4, 'CPAP'), (6, 'PAV')]):
        cls_stats = df[df.cls == cls]
        plt.plot(x_mod+cls_stats[x_lab].values, cls_stats['f1'].values, label=mode, c=cmap[i])

    x_label = {'n': 'Choose First N Breaths', 'ablation': 'Percentage observations removed'}[x_lab]
    y_label = 'F1-score'

    ax.set_xticks(x_mod+cls_stats[x_lab].values, minor=True)
    plt.ylabel(y_label, fontsize=11)
    plt.xlabel(x_label, fontsize=11)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_by_ablation(df, algo, type, x):
    df.loc[df[df.f1.isna()].index, 'f1'] = 0
    df = df[df.ablation != 1]
    x_label = {'ablation': 'Fraction Training Observations Simulated Missing'}[x]
    y_label = 'F1-score'
    colors = ['dark royal blue', 'milk chocolate', 'dull blue', 'carmine', 'deep lavender']
    cmap = sns.color_palette(sns.xkcd_palette(colors))
    for i, (lab, cls) in enumerate([('VC', 0), ('PC', 1), ('PS', 3), ('CPAP', 4), ('PAV', 6)]):
        cls_df = df[df.cls == cls].sort_values(by=x)
        plt.plot(sorted(cls_df[x].values), cls_df['f1'].values, label=lab, c=cmap[i])

    plt.xlabel('Fraction Training Observations Removed')
    plt.ylabel(y_label)
    plt.gca().set_xscale('logit')
    plt.grid(True)
    plt.xlim((min(df.ablation.dropna().values), max(df.ablation.dropna().values)))

    plt.ylim((.15, 1.02))
    plt.yticks(np.arange(0, 1.01, .1))
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_file')
    parser.add_argument('-x', '--x-label', choices=['n', 'ablation'], default='ablation')
    parser.add_argument('-t', '--plot-type', choices=['dtw', 'random', 'size_reduction'], required=True)
    parser.add_argument('-a', '--algo', default='Random Forest')
    parser.add_argument('-n', '--dtw-n', default=16, type=int)
    args = parser.parse_args()

    df = pd.read_pickle(args.pickle_file)
    df.index = range(len(df))
    if args.plot_type == 'size_reduction':
        perform_size_reduction_plot(df, args.x_label)
    elif args.plot_type == 'random':
        plot_by_ablation(df, args.algo, 'Random', args.x_label)
    elif args.plot_type == 'dtw':
        df = df[df.dtw_n == args.dtw_n]
        plot_by_ablation(df, args.algo, 'DTW', args.x_label)


if __name__ == "__main__":
    main()
