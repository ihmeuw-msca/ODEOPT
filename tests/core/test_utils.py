# -*- coding: utf-8 -*-
"""
    test utils
    ~~~~~~~~~~
"""
import numpy as np
import pytest
from odeopt.core import utils


@pytest.mark.parametrize('t_org', [np.arange(5)])
@pytest.mark.parametrize('x_org', [np.arange(5),
                                    np.arange(5).reshape(1, 5)])
@pytest.mark.parametrize('t', [np.arange(0, 5, 0.5)])
@pytest.mark.parametrize('result', [np.minimum(4.0, np.arange(0, 5, 0.5))])
def test_linear_interpolate(t, t_org, x_org, result):
    my_result = utils.linear_interpolate(t, t_org, x_org)
    assert np.allclose(result, my_result.ravel())
    assert my_result.ndim == x_org.ndim


@pytest.mark.parametrize('old_names', ['old', 'new',
                                       ['old', 'new']])
@pytest.mark.parametrize('name_dict', [{'old': 'new'}])
def test_change_names(old_names, name_dict):
    result = utils.change_names(old_names, name_dict)
    if isinstance(result, str):
        assert result == 'new'
    if isinstance(result, list):
        assert all([s == 'new' for s in result])


@pytest.mark.parametrize(('prior', 'result'),
                         [(np.array([0.0, 1.0]), True),
                          (np.array([[0.0, 1.0]]*2), True),
                          (np.array([0.0, -1.0]), False),
                          (np.array([[0.0, -1.0]]*2), False),
                          (None, True),
                          ('gaussian_prior', False)])
def test_is_gaussian_prior(prior, result):
    assert utils.is_gaussian_prior(prior) == result


@pytest.mark.parametrize(('prior', 'result'),
                         [(np.array([0.0, 1.0]), True),
                          (np.array([[0.0, 1.0]]*2), True),
                          (np.array([0.0, -1.0]), False),
                          (np.array([[0.0, -1.0]]*2), False),
                          (None, True),
                          ('uniform_prior', False)])
def test_is_uniform_prior(prior, result):
    assert utils.is_uniform_prior(prior) == result


@pytest.mark.parametrize('prior', [None,
                                   [0.0, np.inf],
                                   [[0.0, np.inf]]*3])
@pytest.mark.parametrize('size', [3])
def test_input_gaussian_prior(prior, size):
    my_prior = utils.input_gaussian_prior(prior, size)
    assert my_prior.shape[0] == size
    assert np.allclose(my_prior[:, 0], 0.0)
    assert np.all(np.isposinf(my_prior[:, 1]))


@pytest.mark.parametrize('prior', [None,
                                   [-np.inf, np.inf],
                                   [[-np.inf, np.inf]]*3])
@pytest.mark.parametrize('size', [3])
def test_input_uniform_prior(prior, size):
    my_prior = utils.input_uniform_prior(prior, size)
    assert my_prior.shape[0] == size
    assert np.all(np.isneginf(my_prior[:, 0]))
    assert np.all(np.isposinf(my_prior[:, 1]))
