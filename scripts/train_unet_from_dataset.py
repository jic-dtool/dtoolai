
import sys
import random
import itertools

import click
import numpy as np

from dtoolcore import DataSet
from imageio import imread, imsave

from dtoolai import ImageMaskGenerator
from unetmodel import get_unet_256


def check_overlays(dataset, overlay_names):

    for overlay_name in overlay_names:
        if overlay_name not in dataset.list_overlay_names():
            print("Error: dataset must have {} overlay".format(overlay_name))
            sys.exit(2)


def train_unet_from_dataset(dataset):

    check_overlays(dataset, ["is_image", "mask_ids"])

    ts = 256
    batch_size = 16
    training_imgen = ImageMaskGenerator(dataset, usetype='training', ts=ts, batch_size=batch_size)
    val_imgen = ImageMaskGenerator(dataset, usetype='validation', ts=ts, batch_size=batch_size)

    training_spe = len(training_imgen) / batch_size
    val_spe = len(val_imgen) / batch_size

    model = get_unet_256()
    print("Got model")
    model.fit_generator(
        training_imgen,
        steps_per_epoch=training_spe,
        epochs=50,
        validation_data=val_imgen,
        validation_steps=val_spe,
        verbose=1
    )

    return model


@click.command()
@click.argument('dataset_uri')
@click.argument('output_model_fpath')
def main(dataset_uri, output_model_fpath):

    dataset = DataSet.from_uri(dataset_uri)

    trained_model = train_unet_from_dataset(dataset)

    trained_model.save(output_model_fpath)


if __name__ == '__main__':
    main()
