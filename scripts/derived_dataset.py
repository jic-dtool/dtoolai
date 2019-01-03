try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import dtoolcore

from ruamel.yaml import YAML

def proto_dataset_from_base_uri(name, base_uri):

    admin_metadata = dtoolcore.generate_admin_metadata(name)
    parsed_base_uri = dtoolcore.utils.generous_parse_uri(base_uri)

    proto_dataset = dtoolcore.generate_proto_dataset(
        admin_metadata=admin_metadata,
        base_uri=dtoolcore.utils.urlunparse(parsed_base_uri)
    )

    proto_dataset.create()

    return proto_dataset


class DerivedDataSet(object):

    def __init__(self, output_base_uri, name):

        self.proto_dataset = proto_dataset_from_base_uri(name, output_base_uri)
        self.readme_dict = {}

    def __enter__(self):

        return self

    def __exit__(self, type, value, traceback):

        yaml = YAML()
        yaml.explicit_start = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        stream = StringIO()
        yaml.dump(self.readme_dict, stream)
        self.proto_dataset.put_readme(stream.getvalue())

        self.proto_dataset.freeze()

    def put_item(self, item_abspath, relpath):

        self.proto_dataset.put_item(item_abspath, relpath)
