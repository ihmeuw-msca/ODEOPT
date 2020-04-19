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
        assert isinstance(data, ODEData)
        assert group in data.groups

        for i in range(self.num_fe):
            effect[i] = self.var_link_fun[i](effect[i])

        param = self.link_fun(
            data.df_by_group(group)[self.col_covs].values.dot(effect)
        )

        return param[:, None]

    def effect2param(self, fe, re, data, groups):
        """Convert the effects to parameters.

        Args:
            fe (numpy.ndarray): Fixed effects.
            re (numpy.ndarray): Random effects.
            data (ODEData): The data object.
            groups (list{any}): list of group definition.

        Returns:
            list{numpy.ndarray}: Parameter by group.
        """
        if not self.use_re:
            assert re.size == 0
        if re.size == 0:
            effect = np.repeat(fe[None, :], len(groups), axis=0)
        else:
            effect = fe + re

        assert effect.shape[0] == len(groups)

        return [
            self._effect2param(effect[i], data, group)
            for i, group in enumerate(groups)
        ]

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


class ParamModel:
    """Parameter Model.
    """
    def __init__(self, single_param_models):
        """Constructor of the ParamModel.

        Args:
            single_param_models (list{SingleParamModel}):
                A list of single parameter models.
        """
        assert isinstance(single_param_models, list)
        assert all([isinstance(model, SingleParamModel)
                    for model in single_param_models])

        self.models = single_param_models
        self.params = [model.name for model in self.models]
        self.num_params = len(self.params)

        self.fe_sizes = np.array([model.num_fe for model in self.models])
        self.re_sizes = np.array([model.num_re for model in self.models])

        self.num_fe = self.fe_sizes.sum()
        self.num_re = self.re_sizes.sum()

        self.fe_idx = utils.sizes_to_indices(self.fe_sizes)
        self.re_idx = utils.sizes_to_indices(self.re_sizes)

    def unpack_optvar(self, x, num_groups):
        """Unpack the optimization variable.

        Args:
            x (numpy.ndarray): Optimization variable.
            num_groups (int): Number of groups.

        Returns:
            list{tupe{np.ndarray, np.ndarray}}:
                List of unpacked fixed and random effects.
        """
        assert x.size == self.num_fe + num_groups*self.num_re

        fe = x[:self.num_fe]
        re = x[self.num_fe:].reshape(num_groups, self.num_re)

        return [
            (fe[self.fe_idx[i]], re[:, self.re_idx[i]])
            for i in range(self.num_params)
        ]

    def optvar2param(self, x, data, groups):
        """Convert optimization variable to parameter.

        Args:
            x (numpy.ndarray): Optimization variable.
            data (ODEData): data object.
            num_groups (int): Number of groups.

        Returns:
            dict{str, np.ndarray}: Parameters by group.
        """
        effect = self.unpack_optvar(x, len(groups))
        params = [
            model.effect2param(effect[i], data, groups)
            for i, model in enumerate(self.models)
        ]
        return {
            group: np.hstack([params[j][i] for j in range(self.num_params)])
            for i, group in enumerate(groups)
        }

    def objective_gprior(self, x, num_groups):
        """Objective from the Gaussian prior.
        """
        effect = self.unpack_optvar(x, num_groups)
        val = np.sum([
            model.objective_gprior(*effect[i])
            for i, model in enumerate(self.models)
        ])

        return val

    def extract_optvar_bounds(self, num_groups):
        """ Extract the bounds from each model.

        Args:
            num_groups (int): Number of groups.

        Returns:
            np.ndarray: bounds for all optimization variable.
        """
        fe_bounds = [model.fe_bounds for model in self.models]
        re_bounds = [model.re_bounds for model in self.models
                     if model.use_re]

        return np.array(fe_bounds + re_bounds*num_groups)
