import dtoolcore

import numpy as np

from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from dtoolai.modelfns import (
    bce_dice_loss,
    weighted_bce_dice_loss,
    dice_coeff
)


def model_from_fpath(model_fpath):

    with CustomObjectScope({
        'bce_dice_loss': bce_dice_loss,
        'dice_coeff': dice_coeff,
        'weighted_bce_dice_loss': weighted_bce_dice_loss,
    }):
        model = load_model(model_fpath)

    return model


def get_idn_by_relpath(dataset, relpath):

    for idn in dataset.identifiers:
        if dataset.item_properties(idn)['relpath'] == relpath:
            return idn

    raise ValueError("Relpath {} not in dataset".format(relpath))


def item_content_abspath_from_relpath(dataset, relpath):

    idn = get_idn_by_relpath(dataset, relpath)

    return dataset.item_content_abspath(idn)


def model_from_dataset_uri(ds_uri):

    dataset = dtoolcore.DataSet.from_uri(ds_uri)
    model_fpath = item_content_abspath_from_relpath(dataset, "model.h5")

    return model_from_fpath(model_fpath)


def find_mask_for_image(im, model, ts=256):

    xdim, ydim, _ = im.shape
    nx = xdim//ts
    ny = ydim//ts

    pad_x = ts * (nx + 1) - xdim
    pad_y = ts * (ny + 1) - ydim

    # impad = np.pad(im, ((0, pad_x), (0, pad_y), (0, 0)), 'constant', constant_values=(0, 0))
    impad = np.pad(im, ((0, pad_x), (0, pad_y), (0, 0)), 'edge')

    tiles = []
    for x in range(nx+1):
        for y in range(ny+1):
            tile = impad[x*ts:(x+1)*ts,y*ts:(y+1)*ts]
            tiles.append(tile)

    X = np.array(tiles).astype(np.float32) / 255

    print(X.min(), X.max())

    results = model.predict(X)

    print(results.min(), results.max())

    n, txdim, tydim, _ = results.shape

    reassembled = reassemble_image(results.reshape(n, txdim, tydim), ny+1)

    cropped = reassembled[:xdim,:ydim]

    rescaled = cropped * (255.0/cropped.max())

    return rescaled.astype(np.uint8)


def grouper(n, iterable):
    args = [iter(iterable)] * n
    return map(list, zip(*args))


def reassemble_image(stack, cols):

    n, _, _ = stack.shape

    nl = [stack[i,:,:] for i in range(n)]

    return np.block(list(grouper(cols, nl)))


def apply_unet_to_image(model, im):

    return find_mask_for_image(im, model)
