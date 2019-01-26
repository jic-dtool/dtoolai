import math
import random

import click

from dtoolcore import DataSet


def annotate_from_relpath(dataset):

    def find_type(identifier):
        relpath = dataset.item_properties(identifier)['relpath']
        stem = relpath.split('/')[0]

        assert stem in ['training', 'validation']

        return stem

    usetype = {idn: find_type(idn) for idn in dataset.identifiers}

    dataset.put_overlay('usetype', usetype)



def annotate_from_tp_selection(dataset):

    validation_tps = [80, 120, 180, 220]
    val_strs = [str(vtp) for vtp in validation_tps]

    def find_type(idn):
        if any(vstr in dataset.item_properties(idn)['relpath'] for vstr in val_strs):
            return 'validation'
        else:
            return 'training'

    usetype = {idn: find_type(idn) for idn in dataset.identifiers}

    dataset.put_overlay('usetype', usetype)


def annotate_by_random_selection(dataset, frac_verify):

    is_image = dataset.get_overlay("is_image")
    mask_lookup = dataset.get_overlay("mask_ids")
    image_idns = [idn for idn in dataset.identifiers if is_image[idn]]
    n_verify = math.floor(frac_verify * len(image_idns))

    verify_image_idns = random.sample(image_idns, n_verify)

    usetype_overlay = {idn: 'training' for idn in dataset.identifiers}

    for idn in verify_image_idns:
        usetype_overlay[idn] = 'validation'
        usetype_overlay[mask_lookup[idn]] = 'validation'

    dataset.put_overlay('usetype', usetype_overlay)

    
@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    dataset = DataSet.from_uri(dataset_uri)
    # annotate_from_tp_selection(dataset)
    # annotate_from_relpath(dataset)
    annotate_by_random_selection(dataset, 0.2)


if __name__ == '__main__':
    main()
