# -*- coding: utf-8 -*-
"""
    test data
    ~~~~~~~~~
"""
import numpy as np
import pandas as pd
import pytest
from odeopt.core.data import ODEData


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


def test_odedata(test_data):
    assert test_data.groups.size == 3
    assert test_data.col_group == 'group'
    assert test_data.col_t == 'time'
    assert test_data.col_components == ['comp1', 'comp2']
    assert test_data.col_covs == ['cov1', 'cov2', 'intercept']


@pytest.mark.parametrize('new_col_names',
                         [{'comp1': 'new_comp1', 'comp2': 'new_comp2'},
                          {'cov1': 'new_cov1', 'cov2': 'new_cov2'},
                          {'group': 'location'},
                          {'time': 't'}])
def test_odedata_rename_cols(test_data, new_col_names):
    test_data.rename_cols(new_col_names)
    if 'comp1' in new_col_names:
        assert test_data.col_components == ['new_comp1', 'new_comp2']
        assert 'new_comp1' in test_data.df
        assert 'new_comp2' in test_data.df
    if 'cov1' in new_col_names:
        assert test_data.col_covs == ['new_cov1', 'new_cov2', 'intercept']
        assert 'new_cov1' in test_data.df
        assert 'new_cov2' in test_data.df
    if 'group' in new_col_names:
        assert test_data.col_group == 'location'
        assert 'location' in test_data.df
    if 'time' in new_col_names:
        assert test_data.col_t == 't'
        assert 't' in test_data.df


@pytest.mark.parametrize('group', ['A', 'B', 'C'])
def test_odedata_df_by_group(test_data, group):
    df_group = test_data.df_by_group(group)
    assert all(df_group[test_data.col_group] == group)
