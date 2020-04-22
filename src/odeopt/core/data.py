# -*- coding: utf-8 -*-
"""
    data
    ~~~~
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

from odeopt.core import utils

@dataclass
class Component:
    col_name: str
    weight: float = 1.0
    se_col_name: str = None 
    new_col_name: str = None


@dataclass
class DataSpecs:
   col_t: str
   col_group: str
   components: List[Component]
   col_covs: List[str] = None


class ODEData:
    """Data used for fitting the ODE parameter.
    """
    def __init__(self, df, data_specs):
        """Constructor of the ODEData.

        Args:
            df (pd.DataFrame): Original data frame file.
            data_specs (DataSpecs): all specs necessary for this data object.
        """
        self.df = df.copy()
        self.data_specs = data_specs
        assert self.data_specs.col_group in self.df
        assert self.data_specs.col_t in self.df
        self.col_group = self.data_specs.col_group
        self.col_t = self.data_specs.col_t
        self.groups = self.df[self.col_group].unique()
        
        assert len(self.data_specs.components) > 0
        self.components = self.data_specs.components
        self.col_components = []
        self.components_weights = []
        self.col_components_se = []
        for component in self.components:
            assert component.col_name in self.df
            if component.weight > 0.0:
                if component.new_col_name is not None:
                    self.df.rename(columns={component.col_name: component.new_col_name}, inplace=True)
                    self.col_components.append(component.new_col_name)
                else:
                    self.col_components.append(component.col_name)
                self.components_weights.append(component.weight)
                self.col_components_se.append(component.se_col_name)

        assert len(self.col_components) > 0
        self.components_weights = [w / sum(self.components_weights) for w in self.components_weights]
        
        self.col_covs = [] if self.data_specs.col_covs is None else self.data_specs.col_covs
        self.df['intercept'] = 1.0
        if 'intercept' not in self.col_covs:
            self.col_covs.append('intercept')
        assert all([name in self.df for name in self.col_covs])
        
        self.df.sort_values([self.col_group, self.col_t], inplace=True)
        self.df = self.df[
            [self.col_group, self.col_t] +
            self.col_components +
            self.col_covs
        ]

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
