# -*- coding: utf-8 -*-
"""
    test_param_model
    ~~~~~~~~~~~~~~~~

    Test Parameter Model.
"""
import numpy as np
import pandas as pd
import pytest
from odeopt.core.data import ODEData
from odeopt.model.param_model import SingleParamModel


@pytest.fixture
def test_df():
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
        'time': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0],
        'comp1': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0],
        'comp2': [0.1, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0],
        'cov1': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0],
        'cov2': [0.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1]
    })
    return df


@pytest.fixture
def test_data(test_df):
    data = ODEData(test_df,
                   'group',
                   'time',
                   ['comp1', 'comp2'],
                   ['cov1', 'cov2'])

    return data


@pytest.fixture
def test_model():
    model = SingleParamModel(
        'alpha',
        ['cov1', 'cov2'],
        link_fun=np.exp,
        var_link_fun=[lambda x: x, lambda x: x],
        use_re = False,
    )
    return model


def test_single_param_model(test_model):
    assert test_model.num_fe == 2
    assert test_model.num_re == 0
    assert test_model.fe_bounds.shape[0] == test_model.num_fe
    assert np.all(np.isneginf(test_model.fe_bounds[:, 0]))
    assert np.all(np.isposinf(test_model.fe_bounds[:, 1]))
    assert np.all(np.isposinf(test_model.fe_gprior[:, 1]))
    assert test_model.re_bounds is None
    assert test_model.re_gprior is None


@pytest.mark.parametrize('group', ['A', 'B', 'C'])
@pytest.mark.parametrize('effect', [np.zeros(2)])
def test_single_param_model_effect2_param_inner(test_model, test_data,
                                                effect, group):
    my_param = test_model._effect2param(effect, test_data, group)
    assert my_param.size == test_data.df_by_group(group).shape[0]

@pytest.mark.parametrize('groups', [['A', 'B', 'C']])
@pytest.mark.parametrize('fe', [np.zeros(2)])
@pytest.mark.parametrize('re', [np.array([]).reshape(3, 0)])
def test_single_param_model_effect2_param_outer(test_model, test_data,
                                                fe, re, groups):
    my_param = test_model.effect2param(fe, re, test_data, groups)
    for i, group in enumerate(groups):
        assert my_param[i].size == test_data.df_by_group(group).shape[0]



@pytest.mark.parametrize('fe', [np.zeros(2)])
@pytest.mark.parametrize('re', [np.array([]),
                                np.array([]).reshape(5, 0)])
def test_single_param_model_objective_gprior(test_model, fe, re):
    my_val = test_model.objective_gprior(fe, re)
    assert my_val == 0.0
