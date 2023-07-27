#!/usr/bin/env python3
"""that performs matrix multiplication"""


def np_slice(matrix, axes={}):
    import numpy as np
    """Slice a matrix along specific axes."""

    x = matrix.copy()
    shape = np.shape(matrix)

    for axis, slice_value in axes.items():
        if isinstance(slice_value, slice):
            start, stop, step = slice_value.start, slice_value.stop, slice_value.step
        else:  # Assume slice_value is a tuple
            if len(slice_value) == 1:
                start, stop, step = slice_value[0], None, None
            elif len(slice_value) == 2:
                start, stop, step = slice_value[0], slice_value[1], None
            else:
                start, stop, step = slice_value
        slices = []
        for i in range(len(shape)):
            if i == axis:
                slices.append(slice(start, stop, step))
            else:
                slices.append(slice(None))

        x = x[tuple(slices)]

    return x
