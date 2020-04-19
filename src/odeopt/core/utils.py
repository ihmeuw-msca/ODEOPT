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


def _change_name(old_name, name_dict):
    """Change the old name to new names.

    Args:
        old_name (str): String of old name.
        name_dict (dict{str, str}): Rules map the old name to new name.

    Returns:
         str: New name.
    """
    assert isinstance(old_name, str)
    assert isinstance(name_dict, dict)

    if old_name in name_dict:
        return name_dict[old_name]
    else:
        return old_name


def change_names(old_names, name_dict):
    """Change the old name to new names.

    Args:
        old_names (str | list{str}): String or a list of string of old names.
        name_dict (dict{str, str}): Rules map the old name to new name.

    Returns:
         new_names (str | list{str}): New names.
    """
    if isinstance(old_names, list):
        new_names = [
            _change_name(name, name_dict)
            for name in old_names
        ]
    else:
        new_names = _change_name(old_names, name_dict)

    return new_names
