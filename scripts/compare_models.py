import os
import json

from pathlib import Path

import click
import dtoolcore

import numpy as np
import matplotlib.pyplot as plt


def get_idn_by_relpath(dataset, relpath):

    for idn in dataset.identifiers:
        if dataset.item_properties(idn)['relpath'] == relpath:
            return idn

    raise ValueError("Relpath {} not in dataset".format(relpath))


def item_content_abspath_from_relpath(dataset, relpath):

    idn = get_idn_by_relpath(dataset, relpath)

    return dataset.item_content_abspath(idn)


def plot_dice(model_histories):

    ncols = 2
    nrows = len(model_histories) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True)

    for n, dsname in enumerate(model_histories):
        history = model_histories[dsname]
        row = n // ncols
        col = n % ncols
        axes[row, col].plot(history["dice_coeff"])
        axes[row, col].plot(history["val_dice_coeff"])
        axes[row, col].set_title(dsname)

    plt.show()


def plot_comparative_end_scores(model_histories):

    to_plot = {
        name: history["val_dice_coeff"][-1]
        for name, history in model_histories.items()
    }
    sorted_keys = sorted(to_plot.keys())
    sorted_vals = [to_plot[k] for k in sorted_keys]
    index = np.arange(len(to_plot))
    labels = [k.split('-')[-1] for k in sorted_keys]

    plt.xlabel("Params")
    plt.ylabel("Val dice coeff")
    plt.title("Validation dice coefficients by parameter combination")
    plt.bar(index, sorted_vals, tick_label=labels)
    # plt.set_xlabels(sorted_keys)
    plt.show()


def compare_models(dirpath):

    model_datasets = (
        dtoolcore.DataSet.from_uri(str(ds_uri))
        for ds_uri in dirpath.iterdir()
    )

    model_histories = {}
    for ds in model_datasets:
        history_fpath = item_content_abspath_from_relpath(ds, 'history.json')
        with open(history_fpath) as fh:
            model_histories[ds.name] = json.load(fh)

    plot_dice(model_histories)


@click.command()
@click.argument('model_dirpath')
def main(model_dirpath):

    compare_models(Path(model_dirpath))
    

if __name__ == "__main__":
    main()
