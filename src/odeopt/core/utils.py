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


def is_gaussian_prior(prior):
    """Check if variable satisfy Gaussian prior format
    Args:
        prior (numpy.ndarray):
            Either one or two dimensional array, with first group refer to mean
            and second group refer to standard deviation.
    Returns:
        bool: True if satisfy condition.
    """
    # check type
    if prior is None:
        return True
    else:
        prior = np.array(prior)
    # check dimension
    if prior.ndim == 1:
        return (prior.size == 2) and (prior[1] > 0.0)
    elif prior.ndim == 2:
        return (prior.shape[1] == 2) and (np.all(prior[:, 1] > 0.0))
    else:
        return False


def is_uniform_prior(prior):
    """Check if variable satisfy uniform prior format
    Args:
        prior (numpy.ndarray):
            Either one or two dimensional array, with first group refer to lower
            bound and second group refer to upper bound.
    Returns:
        bool:
            True if satisfy condition.
    """
    if prior is None:
        return True
    else:
        prior = np.array(prior)
    # check dimension
    if prior.ndim == 1:
        return (prior.size == 2) and (prior[1] >= prior[0] )
    elif prior.ndim == 2:
        return (prior.shape[1] == 2) and (np.all(prior[:, 1] >= prior[:, 0]))
    else:
        return False


def input_gaussian_prior(prior, size):
    """Process the input Gaussian prior
    Args:
        prior (numpy.ndarray | list):
            Either one or two dimensional array, with first group refer to mean
            and second group refer to standard deviation.
        size (int, optional):
            Size the variable, prior related to.
    Returns:
        numpy.ndarray:
            Prior after processing, with shape (2, size), with the first row
            store the mean and second row store the standard deviation.
    """
    if size == 0:
        return None
    assert is_gaussian_prior(prior)
    if prior is None:
        return np.array([[0.0, np.inf]]*size)
    else:
        prior = np.array(prior)
    if prior.ndim == 1:
        return np.repeat(prior[None, :], size, axis=0)
    else:
        assert prior.shape[0] == size
        return prior

def input_uniform_prior(prior, size):
    """Process the input Gaussian prior
    Args:
        prior (numpy.ndarray | list):
            Either one or two dimensional array, with first group refer to mean
            and second group refer to standard deviation.
        size (int, optional):
            Size the variable, prior related to.
    Returns:
        numpy.ndarray:
            Prior after processing, with shape (2, size), with the first row
            store the mean and second row store the standard deviation.
    """
    if size == 0:
        return None
    assert is_uniform_prior(prior)
    if prior is None:
        return np.array([[-np.inf, np.inf]]*size)
    else:
        prior = np.array(prior)
    if prior.ndim == 1:
        return np.repeat(prior[None, :], size, axis=0)
    else:
        assert prior.shape[0] == size
        return prior


def sizes_to_indices(sizes):
    """Converting sizes to corresponding indices.
    Args:
        sizes (numpy.dnarray):
            An array consist of non-negative number.
    Returns:
        list{range}:
            List the indices.
    """
    u_id = np.cumsum(sizes)
    l_id = np.insert(u_id[:-1], 0, 0)

    return [
        np.arange(l, u) for l, u in zip(l_id, u_id)
    ]
