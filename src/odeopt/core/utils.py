# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~

    Utility functions.
"""
import numpy as np


def linear_interpolate(t_target, t_org, x_org):
    """Linearly interpolate the given vector.

    Args:
        t_target (numpy.ndarray):
            Target time points.
        t_org (numpy.ndarray):
            Original time points.
        x_org (numpy.ndarray):
            Original data points regarding `t_org`.

    Returns:
        x_target (numpy.ndarray):
            Interpolated data vector.
    """
    is_vector = x_org.ndim == 1
    if is_vector:
        x_org = x_org[None, :]

    assert t_org.size == x_org.shape[1]

    x_target = np.vstack([
        np.interp(t_target, t_org, x_org[i])
        for i in range(x_org.shape[0])
    ])

    if is_vector:
        return x_target.ravel()
    else:
        return x_target
