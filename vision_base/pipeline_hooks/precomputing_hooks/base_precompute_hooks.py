import numpy as np


def skew(T):
    return np.array(
        [[0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]]
    )

class BasePrecomputeHook(object):
    """
        Precomputing functions do not have input/output arguments.
        But they can have initialization parameters.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass
