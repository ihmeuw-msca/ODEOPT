# -*- coding: utf-8 -*-
"""
    test data
    ~~~~~~~~~
"""
import numpy as np
import pandas as pd
import pytest
from odeopt.core.data import ODEData


class TestODEData:

    @pytest.fixture
    def df(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
            'time': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0],
            'comp1': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0],
            'comp2': [0.1, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0],
            'cov1': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0],
            'cov2': [0.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1]
        })
        return df

    def generate_odedata_instance(self, df, components_weights=None):
        data = ODEData(
            df,
            'group',
            'time',
            ['comp1', 'comp2'],
            components_weights=components_weights,
            col_covs=['cov1', 'cov2'],
        )

        return data

    def test_odedata(self, df):
        odedata = self.generate_odedata_instance(df)
        assert odedata.groups.size == 3
        assert odedata.col_group == 'group'
        assert odedata.col_t == 'time'
        assert odedata.col_components == ['comp1', 'comp2']
        assert odedata.components_weights == [0.5, 0.5]
        assert odedata.col_covs == ['cov1', 'cov2', 'intercept']

    def test_odedata_with_weights(self, df):

        with pytest.raises(AssertionError):
            self.generate_odedata_instance(df, components_weights=[1.0, 0.0, 2.0])
        
        with pytest.raises(AssertionError):
            self.generate_odedata_instance(df, components_weights=[-1.0, 0.0])

        odedate = self.generate_odedata_instance(df, components_weights=[0.5, 0.0])
        assert odedate.col_components == ['comp1']
        assert odedate.components_weights == [1.0]

        odedate = self.generate_odedata_instance(df, components_weights=[0.5, 1.5])
        assert odedate.col_components == ['comp1', 'comp2']
        assert odedate.components_weights == [0.25, 0.75]

    @pytest.mark.parametrize('new_col_names',
                            [{'comp1': 'new_comp1', 'comp2': 'new_comp2'},
                            {'cov1': 'new_cov1', 'cov2': 'new_cov2'},
                            {'group': 'location'},
                            {'time': 't'}])
    def test_odedata_rename_cols(self, df, new_col_names):
        odedate = self.generate_odedata_instance(df)
        odedate.rename_cols(new_col_names)
        if 'comp1' in new_col_names:
            assert odedate.col_components == ['new_comp1', 'new_comp2']
            assert 'new_comp1' in odedate.df
            assert 'new_comp2' in odedate.df
        if 'cov1' in new_col_names:
            assert odedate.col_covs == ['new_cov1', 'new_cov2', 'intercept']
            assert 'new_cov1' in odedate.df
            assert 'new_cov2' in odedate.df
        if 'group' in new_col_names:
            assert odedate.col_group == 'location'
            assert 'location' in odedate.df
        if 'time' in new_col_names:
            assert odedate.col_t == 't'
            assert 't' in odedate.df


    @pytest.mark.parametrize('group', ['A', 'B', 'C'])
    def test_odedata_df_by_group(self, df, group):
        odedata = self.generate_odedata_instance(df)
        df_group = odedata.df_by_group(group)
        assert all(df_group[odedata.col_group] == group)
