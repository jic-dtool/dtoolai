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


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    dataset = DataSet.from_uri(dataset_uri)

    annotate_from_relpath(dataset)


if __name__ == '__main__':
    main()
