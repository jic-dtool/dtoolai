import os
import shutil
import tempfile

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

    def __init__(self, output_base_uri, name, source_ds=None):

        self.proto_dataset = proto_dataset_from_base_uri(name, output_base_uri)
        self.readme_dict = {}

        if source_ds is not None:
            self.readme_dict['source_ds_name'] = source_ds.name
            self.readme_dict['source_ds_uri'] = source_ds.uri
            self.readme_dict['source_ds_uuid'] = source_ds.uuid

    def __enter__(self):

        self.tmpdir = tempfile.mkdtemp()
        self.to_stage = []

        return self

    def _create_readme(self):

        yaml = YAML()
        yaml.explicit_start = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        stream = StringIO()
        yaml.dump(self.readme_dict, stream)
        self.proto_dataset.put_readme(stream.getvalue())

    def __exit__(self, type, value, traceback):

        for abspath, relpath in self.to_stage:
            self.proto_dataset.put_item(abspath, relpath)

        self._create_readme()

        self.proto_dataset.freeze()

        shutil.rmtree(self.tmpdir)

    def put_item(self, item_abspath, relpath):

        self.proto_dataset.put_item(item_abspath, relpath)

    def staging_fpath(self, relpath):
        # TODO - work with full path structure

        staging_abspath = os.path.join(self.tmpdir, relpath)
        self.to_stage.append((staging_abspath, relpath))

        return staging_abspath

        
