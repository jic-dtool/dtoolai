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

    classes = dataset.get_overlay("classes")
    all_values = list(classes.values())
    counts_by_class = {cls: all_values.count(cls) for cls in set(all_values)}

    print('\n'.join(
        "{:<20} {}".format(cls, cnt) for cls, cnt in counts_by_class.items()
    ))

    n_verify = math.floor(frac_verify * len(dataset.identifiers))
    verify_idns = random.sample(dataset.identifiers, n_verify)

    usetype_overlay = {idn: 'training' for idn in dataset.identifiers}
    for idn in verify_idns:
        usetype_overlay[idn] = 'validation'

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
