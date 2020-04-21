# -*- coding: utf-8 -*-
"""
    ode_model
    ~~~~~~~~~
"""
import numpy as np
from scipy.optimize import minimize
from .init_model import InitModel
from .param_model import ParamModel
from odeopt.ode import ODESys
from odeopt.core import ODEData


class ODEModel:
    """ODE Model.
    """
    def __init__(self, data, ode_sys, init_model, param_model):
        """Constructor function for ODEModel.

        Args:
            data (ODEData): The data object.
            ode_sys (ODESys): The ODE system object.
            init_model (InitModel): Initial condition Model.
            param_model (ParamModel): Parameter Model.
        """
        assert isinstance(data, ODEData)
        assert isinstance(ode_sys, ODESys)
        assert isinstance(init_model, InitModel)
        assert isinstance(param_model, ParamModel)

        # check the compatibility
        assert all([c in ode_sys.components for c in data.col_components])
        assert ode_sys.params == param_model.params
        assert ode_sys.components == init_model.components

        self.data = data
        self.ode_sys = ode_sys
        self.param_model = param_model
        self.init_model = init_model

        self.num_groups = len(self.data.groups)

        self.init_var_size = \
            init_model.num_fe + init_model.num_re*self.num_groups
        self.param_var_size = \
            param_model.num_fe + param_model.num_re*self.num_groups
        self.var_size = self.init_var_size + self.param_var_size

        self.optvar_bounds = np.vstack([
            self.init_model.extract_optvar_bounds(self.num_groups),
            self.param_model.extract_optvar_bounds(self.num_groups)
        ])

        self.result = None
        self.result_inits = None
        self.result_params = None

    def objective_by_group(self, inits, params, group):
        """Objective for a specific group.

        Args:
            inits (dict{str, np.ndarray}): Initial condition for each group.
            params (dict{str, np.ndarray}): Parameters for each group.
            group (any): group definition.

        Returns:
            float: objective value.
        """
        prediction = self.predict(
            self.data.df_by_group(group)[self.data.col_t],
            inits=inits,
            params=params,
            group=group
        )

        val = 0.0
        for component, weight in zip(self.data.col_components, self.data.components_weights):
            observation = self.data.df_by_group(group)[component]
            residual = observation - prediction[
                self.ode_sys.components_id[component]]
            val += 0.5*np.sum(residual**2) * weight

        return val

    def predict(self, t, inits=None, params=None, group=None):
        """Prediction from the simulation of ODE System.

        Args:
            t (np.ndarray): time points where want to predict.
            inits (dict | None, optional):
                Dictionary of initial condition. When it is None, use the final
                result initial condition.
            params (dict | None, optional):
                Dictionary of the parameters. When it is None, use the final
                result parameters.
            group (any | None, optional):
                Which group want to be predicted. If None, returns prediction
                for all groups as a dictionary.

        Returns:
            np.ndarray | dict{any, np.ndarray}:
                The prediction for a single group or all groups.
        """
        if inits is None:
            inits = self.result_inits
        if params is None:
            params = self.result_params

        if inits is None or params is None:
            RuntimeError('Fit the algorithm first or pass in initial condition'
                         'or parameters.')

        if group is not None:
            assert group in self.data.groups
            prediction = self.ode_sys.simulate(
                t,
                inits[group],
                self.data.df_by_group(group)[self.data.col_t],
                params[group],
            )
        else:
            prediction = {
                group: self.ode_sys.simulate(
                    t,
                    inits[group],
                    self.data.df_by_group(group)[self.data.col_t],
                    params[group],
                )
                for group in self.data.groups
            }
        return prediction

    def objective(self, x):
        """Objective function.

        Args:
            x (numpy.ndarray): Optimization variable.

        Returns:
            float: objective value.
        """
        x_init = x[:self.init_var_size]
        x_param = x[self.init_var_size:]

        inits = self.init_model.optvar2param(x_init,
                                             self.data,
                                             self.data.groups)
        params = self.param_model.optvar2param(x_param,
                                               self.data,
                                               self.data.groups)

        val = np.sum([
            self.objective_by_group(inits, params, group)
            for group in self.data.groups
        ])

        val += self.param_model.objective_gprior(x_param, self.num_groups)
        val += self.init_model.objective_gprior(x_init, self.num_groups)

        return val

    def gradient(self, x):
        """Gradient function.
        Args:
            x (numpy.ndarray):
                Model parameters.
        Returns:
            numpy.ndarray:
                Gradient w.r.t. the model parameters.
        """
        finfo = np.finfo(float)
        step = finfo.tiny / finfo.eps
        x_c = x + 0j
        grad = np.zeros(x.size)
        for i in range(x.size):
            x_c[i] += step*1j
            grad[i] = self.objective(x_c).imag/step
            x_c[i] -= step*1j

        return grad

    def fit_model(self, x0, options=None):
        """Fit the model, including initial condition and parameter.

        Args:
            x0 (numpy.ndarray):
                Initial guess for the optimization variable.
            options (None | dict):
                Optimization solver options.
        """
        result = minimize(
            fun=self.objective,
            x0=x0,
            jac=self.gradient,
            method='L-BFGS-B',
            bounds=self.optvar_bounds,
            options=options
        )

        self.result = result
        self.result_inits = self.init_model.optvar2param(
            result.x[:self.init_var_size], self.data, self.data.groups)
        self.result_params = self.param_model.optvar2param(
            result.x[self.init_var_size:], self.data, self.data.groups)
