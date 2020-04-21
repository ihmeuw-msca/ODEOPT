# -*- coding: utf-8 -*-
"""
    data
    ~~~~
"""
import numpy as np
import pandas as pd
from odeopt.core import utils


class ODEData:
    """Data used for fitting the ODE parameter.
    """
    def __init__(
        self, 
        df, 
        col_group, 
        col_t, 
        col_components, 
        components_weights=None, 
        col_covs=None, 
        new_col_names=None,
    ):
        """Constructor of the ODEData.

        Args:
            df (pd.DataFrame): Original data frame file.
            col_group (str): Name of the group column.
            col_t (str): Name of the time column.
            col_components (list{str}): Names of the component columns.
            col_covs (list{str} | None, optional): Names of the covariates.
            new_col_names (dict{str, str}): Rename the columns.
        """
        self.df_original = df.copy()
        self.col_group = col_group
        self.col_t = col_t
        
        assert len(col_components) > 0
        if components_weights is None:
            self.col_components = col_components
            self.components_weights = [1.0 / len(col_components)] * len(col_components)
        else:
            assert len(col_components) == len(components_weights)
            assert sum(components_weights) > 0.0
            self.col_components = [col for (col, w) in zip(col_components, components_weights) if w > 0.0]
            self.components_weights = [w for w in components_weights if w > 0.0]
            self.components_weights = [w / sum(self.components_weights) for w in self.components_weights]
        
        self.col_covs = [] if col_covs is None else col_covs

        # add intercept as default covariates
        df['intercept'] = 1.0
        if 'intercept' not in self.col_covs:
            self.col_covs.append('intercept')

        assert col_group in df
        assert col_t in df
        assert all([name in df for name in self.col_components])
        assert all([name in df for name in self.col_covs])
        self.df = df[[self.col_group, self.col_t] +
                     self.col_components +
                     self.col_covs].copy()
        self.df.sort_values([col_group, col_t], inplace=True)
        self.rename_cols(new_col_names)

        self.groups = self.df[col_group].unique()

    def rename_cols(self, new_col_names):
        """Rename the columns in the data set.

        Args:
            new_col_names (dict{str, str}): New column names.
        """
        if new_col_names is not None:
            assert all([name in self.df for name in new_col_names])
            self.df.rename(columns=new_col_names, inplace=True)
            self.col_group = utils.change_names(self.col_group, new_col_names)
            self.col_t = utils.change_names(self.col_t, new_col_names)
            self.col_components = utils.change_names(self.col_components,
                                                     new_col_names)
            self.col_covs = utils.change_names(self.col_covs, new_col_names)

    def df_by_group(self, group):
        """Divide data by group.

        Args:
            group (any): Group id in the data frame.

        Returns:
            pd.DataFrame: The corresponding data frame.
        """
        assert group in self.groups
        return self.df[self.df[self.col_group] == group]
