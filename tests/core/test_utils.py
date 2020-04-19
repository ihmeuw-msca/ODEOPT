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
