
def identifiers_where_overlay_is_true(dataset, overlay_name):
    overlay = dataset.get_overlay(overlay_name)

    selected = [identifier
                for identifier in dataset.identifiers
                if overlay[identifier]]

    return selected

