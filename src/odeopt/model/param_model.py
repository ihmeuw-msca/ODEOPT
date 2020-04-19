# -*- coding: utf-8 -*-
"""
    param_model
    ~~~~~~~~~~~

    Parameter Model.
"""
import numpy as np
from odeopt.core import utils
from odeopt.core.data import ODEData


class SingleParamModel:
    """Single Parameter Model.
    """
    def __init__(self, name, col_covs, link_fun, var_link_fun,
                 use_re=False,
                 fe_bounds=None,
                 re_bounds=None,
                 fe_gprior=None,
                 re_gprior=None):
        """Constructor of the SingleParamModel.

        Args:
            name (str): Name of the parameter.
            col_covs (list{str}): List of covariates.
            link_fun(callable): Link function for the parameters.
            var_link_fun (list{callable}): Variable link function.
            use_re (bool, optional): If use random effects or not.
            fe_bounds (list{list{float}}): Bounds for the fixed effects.
            re_bounds (list{list{float}}): Bounds for the random effects.
            fe_gprior (list{list{float}}): Gaussian prior for the fixed effects.
            re_gprior (list{list{float}}):
                Gaussian prior for the random effects.
        """
        assert isinstance(name, str)
        assert isinstance(col_covs, list)
        assert len(col_covs) != 0
        assert all([isinstance(s, str) for s in col_covs])
        assert callable(link_fun)
        assert isinstance(var_link_fun, list)
        assert all([callable(f) for f in var_link_fun])

        self.name = name
        self.col_covs = col_covs
        self.link_fun = link_fun
        self.var_link_fun = var_link_fun

        self.use_re = use_re
        self.num_fe = len(self.col_covs)
        self.num_re = self.num_fe if use_re else 0
        assert len(var_link_fun) == self.num_fe

        self.fe_bounds = utils.input_uniform_prior(fe_bounds, self.num_fe)
        self.re_bounds = utils.input_uniform_prior(re_bounds, self.num_re)
        self.fe_gprior = utils.input_gaussian_prior(fe_gprior, self.num_fe)
        self.re_gprior = utils.input_gaussian_prior(re_gprior, self.num_re)

    def effect2param(self, effect, data, group):
        """Convert effect to parameter.

        Args:
            effect (numpy.ndarray): Effect for a specific group.
            data (ODEData): The data object.
            group (any): The group we want to compute the parameter for.

        Returns:
            numpy.ndarray: Corresponding parameter.
        """
        assert len(effect) == self.num_fe
        assert isinstance(data, ODEData)
        assert group in data.groups

        for i in range(self.num_fe):
            effect[i] = self.var_link_fun[i](effect[i])

        param = self.link_fun(
            data.df_by_group(group)[self.col_covs].values.dot(effect)
        )

        return param

    def objective_gprior(self, fe, re):
        """Objective from the Gaussian prior.

        Args:
            fe (numpy.ndarray): 1D array for fixed effects.
            re (numpy.ndarray): 1D or 2D array for random effects.

        Returns:
            float: Number for the objective.
        """
        assert fe.ndim == 1
        if re.ndim == 1:
            re = re[None, :]

        assert fe.size == self.num_fe
        assert re.shape[1] == self.num_re

        val = 0.5*np.sum(((fe - self.fe_gprior[:, 0])/self.fe_gprior[:, 1])**2)
        if self.re_gprior is not None:
            val += 0.5*np.sum(
                ((re - self.re_gprior[:, 0])/self.re_gprior[:, 1])**2
            )

        return val
