
def identifiers_where_overlay_is_true(dataset, overlay_name):
    overlay = dataset.get_overlay(overlay_name)

    selected = [identifier
                for identifier in dataset.identifiers
                if overlay[identifier]]

    return selected


def tile_generator(im, ts=256):

    rows, cols = im.shape[0], im.shape[1]

    nr = rows//ts
    nc = cols//ts

    for r in range(nr):
        for c in range(nc):
            tile = im[r*ts:(r+1)*ts,c*ts:(c+1)*ts]
            yield tile
