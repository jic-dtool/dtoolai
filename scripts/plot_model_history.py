"""Plot the history of a keras model from a dataset."""

import json

import click
import dtoolcore

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_idn_by_relpath(dataset, relpath):

    for idn in dataset.identifiers:
        if dataset.item_properties(idn)['relpath'] == relpath:
            return idn

    raise ValueError("Relpath {} not in dataset".format(relpath))


def item_content_abspath_from_relpath(dataset, relpath):

    idn = get_idn_by_relpath(dataset, relpath)

    return dataset.item_content_abspath(idn)


def plot_history_from_dataset(input_ds):

    history_fpath = item_content_abspath_from_relpath(input_ds, "history.json")


    with open(history_fpath) as fh:
        history = json.load(fh)

    plt.plot(history["dice_coeff"])
    plt.plot(history["val_dice_coeff"])
    plt.title("model performance")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    input_ds = dtoolcore.DataSet.from_uri(dataset_uri)

    plot_history_from_dataset(input_ds)


if __name__ == "__main__":
    main()
