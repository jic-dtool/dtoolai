
import numpy as np

from imageio import imread

from keras.preprocessing.image import ImageDataGenerator

from dtoolai.utils import identifiers_where_overlay_is_true


def tile_generator(im, ts=256):

    rows, cols = im.shape[0], im.shape[1]

    nr = rows//ts
    nc = cols//ts

    for r in range(nr):
        for c in range(nc):
            tile = im[r*ts:(r+1)*ts,c*ts:(c+1)*ts]
            yield tile


def get_imids_from_dataset_and_usetype(dataset, usetype=None):

    selected_imids = identifiers_where_overlay_is_true(dataset, "is_image")

    if usetype is not None:
        usetype_overlay = dataset.get_overlay("usetype")
        allowed_usetypes = set(usetype_overlay.values())
        if usetype not in allowed_usetypes:
            raise ValueError("Use type {} not in overlay".format(usetype))
        selected_imids = [
            idn
            for idn in selected_imids
            if usetype_overlay[idn] == usetype
        ]

    return selected_imids


def dataset_tile_generator(dataset, usetype=None, ts=256):

    mask_ids = dataset.get_overlay("mask_ids")

    selected_imids = get_imids_from_dataset_and_usetype(dataset, usetype)

    for imid in selected_imids:
        im = imread(dataset.item_content_abspath(imid))
        mask = imread(dataset.item_content_abspath(mask_ids[imid]))

        imgen = tile_generator(im, ts=ts)
        maskgen = tile_generator(mask, ts=ts)
        for itile, mtile in zip(imgen, maskgen):
            yield itile, mtile


class ImageMaskGenerator(object):

    def __init__(self, dataset, usetype=None, ts=256, batch_size=1):

        self.dataset = dataset
        self.ts = ts
        self.usetype = usetype
        self.batch_size = batch_size
        self.normalise = True

        self.tgen = dataset_tile_generator(dataset, usetype, ts)

        data_gen_args = dict(
            rotation_range=90.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2
        )

        self.image_datagen = ImageDataGenerator(**data_gen_args)

    def __iter__(self):
        return self

    def __len__(self):

        count = 0
        for imid in get_imids_from_dataset_and_usetype(self.dataset, self.usetype):
            im = imread(self.dataset.item_content_abspath(imid))
            rows, cols = im.shape[0], im.shape[1]

            nr = rows//self.ts
            nc = cols//self.ts

            count += nr * nc

        return count

    def next(self):
        return self.__next__()

    def __next__(self):

        itiles = []
        mtiles = []

        for i in range(self.batch_size):
            try:
                itile, mtile = next(self.tgen)

            except StopIteration:
                self.tgen = dataset_tile_generator(self.dataset, self.ts)
                itile, mtile = next(self.tgen)

            itiles.append(itile)
            mtiles.append(mtile)

        X = np.array(itiles)
        Y = np.array(mtiles).reshape(self.batch_size, self.ts, self.ts, 1)

        if self.normalise:
            X /= 255
            Y /= 255

        return X, Y
