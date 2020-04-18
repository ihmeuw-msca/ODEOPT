# -*- coding: utf-8 -*-
"""
    sysode
    ~~~~~~

    System of ODE.
"""
import numpy as np
from scipy.integrate import solve_ivp


class ODESys:
    """System of ODE.
    """
    def __init__(self, system, params, component_names):
        """Constructor of the SysODE.

        Args:
            system (callable): Derivatives of the system.
            params (dict{str, callable}):
                Dictionary with parameter name as the key and parameter as the
                function of time.
            component_names (list{str} , optional):
                The name of the components.
        """
        assert callable(system)
        assert isinstance(params, dict)
        assert isinstance(component_names, list)

        self.system = system
        self.params = params
        self.param_names = list(self.params.keys())
        self.component_names = component_names

        self.num_params = len(self.params)
        self.num_components = len(self.component_names)

    def update_params(self, params):
        """Update the parameters.

        Args:
            params (dict{str, callable}): Updated parameters.
        """
        assert all([name in self.param_names for name in params])
        self.params.update(params)

    def solve_ode(self, t, init_cond):
        """Solve the ODE by given time and initial condition.

        Args:
            t (numpy.ndarray):
                Time vector where want to evaluate the components.
            init_cond (numpy.ndarray | dict):
                Array of initial condition or dictionary with component names
                and initial conditions.

        Returns:
            dict{str, numpy.ndarray}:
                Result of the initial value problem.
        """
        assert len(init_cond) == self.num_components
        assert hasattr(t, '__iter__')
        t = np.sort(np.unique(np.array(t)))
        assert t.size >= 2

        if isinstance(init_cond, dict):
            init_cond = np.array([
                init_cond[name]
                for name in self.component_names
            ])

        soln = solve_ivp(fun=lambda tv, yv: self.system(tv, yv, self.params),
                         t_span=[np.min(t), np.max(t)],
                         y0=init_cond,
                         t_eval=t)

        result = {'time': soln.t}
        result.update({
            self.component_names[i]: soln.y[i]
            for i in range(self.num_components)
        })
        return result
