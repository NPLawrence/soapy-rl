import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "processed" / "all_data.pkl"
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_pickle(data_path)

    runs_df['eval_mean'] = runs_df['eval'].apply(lambda x: 100*np.mean(x))
    runs_df['eval_td_error'] = runs_df['eval_td_error'].apply(lambda x: np.mean(x))
    runs_df['train_td_error'] = runs_df['train_td_error'].apply(lambda x: np.mean(x))
    runs_df['success'] = runs_df['eval'].apply(lambda x: np.mean(x[-10:-1] > 0.3))

    data_soap = runs_df[(runs_df['optimizer']=='soap') & (runs_df['learning_rate']==0.0001)]
    data_adam = runs_df[(runs_df['optimizer']=='adamw') & (runs_df['learning_rate']==0.00001)]
    data = pd.concat([data_soap, data_adam], ignore_index=True)

    sns.set(palette='Set2', style='ticks')

    SMALL_SIZE = 12
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    #  fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=2.5)

    params = {
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)

    fig, ax = plt.subplots(2, sharex=True, layout="constrained")

    ax[0].set_xscale('log')
    ax[0].set_ylabel('Mean cumulative reward \n over training')
    ax[1].set_ylabel('Success rate')

    ax[1].set_xlabel('Dataset size')

    def min_max_error(x):
        return x.min(), x.max()

    sns.lineplot(ax=ax[0], data=data, x='num_data', y="eval_mean", hue='optimizer', estimator=np.median, errorbar=min_max_error, style='optimizer')


    sns.lineplot(ax=ax[1],  data=data, x='num_data', y="success", hue='optimizer', estimator=np.median, errorbar=min_max_error, style='optimizer', legend=False)

    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles=handles, labels=['SOAP', 'AdamW'], title=None) 
    sns.move_legend(ax[0], "lower center", title=None, bbox_to_anchor=(0.5, 1), ncol=2)

    plt.tick_params(axis='both', which='major', top=True, right=True, bottom=True, left=True, length=5, width=1)

    plt.savefig(figures_dir / 'eval.png')
    plt.savefig(figures_dir / 'eval.pdf')


if __name__ == "__main__":
    main()
