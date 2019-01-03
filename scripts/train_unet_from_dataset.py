import os
import sys

import json

import click

from dtoolcore import DataSet

from dtoolai import ImageMaskGenerator

from unetmodel import get_unet_256
from derived_dataset import DerivedDataSet

import shutil
import tempfile
from contextlib import contextmanager

from parameters import Parameters

@contextmanager
def temp_dir_context():

    temp_dir = tempfile.mkdtemp()

    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
    

def check_overlays(dataset, overlay_names):

    for overlay_name in overlay_names:
        if overlay_name not in dataset.list_overlay_names():
            print("Error: dataset must have {} overlay".format(overlay_name))
            sys.exit(2)


def train_unet_from_dataset(dataset, params):

    check_overlays(dataset, ["is_image", "mask_ids"])

    ts = params.tile_size
    batch_size = params.batch_size
    training_imgen = ImageMaskGenerator(dataset, usetype='training', ts=ts, batch_size=batch_size)
    val_imgen = ImageMaskGenerator(dataset, usetype='validation', ts=ts, batch_size=batch_size)

    training_spe = len(training_imgen) / batch_size
    val_spe = len(val_imgen) / batch_size

    model = get_unet_256(
        bn=params.batchnorm,
        do=params.dropout,
        do_frac=params.dropout_frac,
        cl=params.crosslinks
    )
    print("Got model")
    history = model.fit_generator(
        training_imgen,
        steps_per_epoch=training_spe,
        epochs=50,
        validation_data=val_imgen,
        validation_steps=val_spe,
        verbose=1
    )

    return model, history


def train_and_save_results(input_ds, output_ds):

    params = Parameters()
    params.parameter_dict['model_name'] = 'UNet256'
    params.parameter_dict['dropout'] = True
    params.parameter_dict['dropout_frac'] = 0.2
    params.parameter_dict['batchnorm'] = True
    params.parameter_dict['crosslinks'] = False
    params.parameter_dict['tile_size'] = 256
    params.parameter_dict['batch_size'] = 16

    output_ds.readme_dict['parameters'] = params.parameter_dict

    with temp_dir_context() as tmp_dir:
        trained_model, history = train_unet_from_dataset(input_ds, params)

        history_fpath = os.path.join(tmp_dir, 'history.json')
        with open(history_fpath, 'w') as fh:
            json.dump(history.history, fh)

        model_fpath = os.path.join(tmp_dir, 'model.h5')
        trained_model.save(model_fpath)

        output_ds.put_item(history_fpath, 'history.json')
        output_ds.put_item(model_fpath, 'model.h5')



@click.command()
@click.argument('input_dataset_uri')
@click.argument('output_fpath')
def main(input_dataset_uri, output_fpath):

    output_base_uri = os.path.dirname(output_fpath)
    output_name = os.path.basename(output_fpath)

    input_ds = DataSet.from_uri(input_dataset_uri)

    readme = {}
    readme['input_ds_uri'] = input_dataset_uri
    readme['input_ds_name'] = input_ds.name
    readme['input_ds_uuid'] = input_ds.uuid
    with DerivedDataSet(output_base_uri, output_name) as output_ds:
        output_ds.readme_dict = readme

        train_and_save_results(input_ds, output_ds)


if __name__ == '__main__':
    main()
