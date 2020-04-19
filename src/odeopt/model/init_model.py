# -*- coding: utf-8 -*-
"""
    init_model
    ~~~~~~~~~~

    Initial Condition Model.
"""
import numpy as np
from odeopt.core import utils
from odeopt.core.data import ODEData
from .param_model import SingleParamModel
from .param_model import ParamModel


class SingleInitModel(SingleParamModel):
    """Initial condition model for each component.
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, ['intercept'], *args, **kwargs)

    def _effect2param(self, effect, data, group):
        """Convert effect to parameter.

        Args:
            effect (numpy.ndarray): Effect for a specific group.
            data (ODEData): The data object.
            group (any): The group we want to compute the parameter for.

        Returns:
            numpy.ndarray: Corresponding parameter.
        """
        assert len(effect) == self.num_fe
        return np.array([self.link_fun(self.var_link_fun[0](effect[0]))])


class InitModel(ParamModel):
    """Initial condition model.
    """
    def __init__(self, single_init_models):
        """Constructor of the ParamModel.

        Args:
            single_param_models (list{SingleParamModel}):
                A list of single parameter models.
        """
        super().__init__(single_init_models)
        self.components = self.params
