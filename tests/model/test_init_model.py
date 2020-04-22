# -*- coding: utf-8 -*-
"""
    test_init_model
    ~~~~~~~~~~~~~~~
"""
import numpy as np
import pandas as pd
import pytest
from odeopt.core.data import ODEData, DataSpecs, Component
from odeopt.model import SingleInitModel
from odeopt.model import InitModel

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
    comps = [Component(col_name='comp1'), Component(col_name='comp2')]
    data_specs = DataSpecs(col_t='time', col_group='group', components=comps, col_covs=['cov1', 'cov2'])
    data = ODEData(test_df, data_specs)
    return data


@pytest.fixture
def test_model():
    model = SingleInitModel(
        'alpha',
        link_fun=np.exp,
        use_re=False,
    )
    return model

def test_single_init_model(test_model):
    assert test_model.num_fe == 1
    assert test_model.num_re == 0
    assert test_model.fe_bounds.shape[0] == test_model.num_fe
    assert np.all(np.isneginf(test_model.fe_bounds[:, 0]))
    assert np.all(np.isposinf(test_model.fe_bounds[:, 1]))
    assert np.all(np.isposinf(test_model.fe_gprior[:, 1]))
    assert test_model.re_bounds is None
    assert test_model.re_gprior is None


@pytest.mark.parametrize('group', ['A', 'B', 'C'])
@pytest.mark.parametrize('effect', [np.zeros(1)])
def test_single_init_model_effect2_param_inner(test_model, test_data,
                                                effect, group):
    my_param = test_model._effect2param(effect, test_data, group)
    assert my_param.size == 1

@pytest.mark.parametrize('groups', [['A', 'B', 'C']])
@pytest.mark.parametrize('fe', [np.zeros(1)])
@pytest.mark.parametrize('re', [np.array([]).reshape(3, 0)])
def test_single_init_model_effect2_param_outer(test_model, test_data,
                                                fe, re, groups):
    my_param = test_model.effect2param(fe, re, test_data, groups)
    for i, group in enumerate(groups):
        assert my_param[i].size == 1


def test_param_model(test_model):
    model = InitModel([test_model])
    assert model.num_params == 1
    assert model.num_fe == 1
    assert model.num_re == 0
    assert np.allclose(model.fe_idx[0], np.arange(0, 1))
    assert np.allclose(model.re_idx[0], np.arange(0, 0))
    assert model.components == model.params
