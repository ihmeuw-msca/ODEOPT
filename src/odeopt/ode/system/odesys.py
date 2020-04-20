# -*- coding: utf-8 -*-
"""
    sysode
    ~~~~~~

    System of ODE.
"""
import numpy as np
from odeopt.ode import ODESolver
from odeopt.ode import ForwardEuler

class ODESys:
    """System of ODE.
    """
    def __init__(self, system, params, components,
                 solver_class=ForwardEuler, solver_dt=1.0):
        """Constructor of the SysODE.

        Args:
            system (callable): Derivatives of the system.
            params (list{str}): Names of the parameters
            component_names (list{str}): Name of the components.
            solver_class (type, optional):
                ODE solver class, default use the ForwardEuler method.
            solver_dt (float, optional): Default step size of the ODE solver.
        """
        assert callable(system)
        assert hasattr(params, '__iter__')
        assert hasattr(components, '__iter__')
        params = list(params)
        components = list(components)
        assert all([isinstance(name, str) for name in params])
        assert all([isinstance(name, str) for name in components])

        self.system = system
        self.params = params
        self.components = components

        self.params_id = {
            name: i for i, name in enumerate(self.params)
        }
        self.components_id = {
            name: i for i, name in enumerate(self.components)
        }

        self.num_params = len(self.params)
        self.num_components = len(self.components)

        assert issubclass(solver_class, ODESolver)
        assert solver_dt > 0.0
        self.solver_class = solver_class
        self.solver_dt = solver_dt
        self.solver = self.solver_class(self.system, self.solver_dt)

    def update_solver(self, solver_class=None, solver_dt=None):
        """Update the ODE Solver.

        Args:
            solver_class (type | None, optional):
                Alternative solver class.
            solver_dt (float | None, optional):
                Alternative step size.
        """
        if solver_class is not None:
            self.solver_class = solver_class
            self.solver = self.solver_class(self.system, self.solver_dt)

        if solver_dt is not None:
            self.solver_dt = solver_dt
            self.solver.dt = self.solver_dt

    def simulate(self, t, t_params, params, init_cond):
        """Solve the ODE by given time and initial condition.

        Args:
            t (numpy.ndarray):
                Time points where we evaluate the system of ODE. Assume to be
                sorted.
            t_params (numpy.ndarray):
                Time stamp for the parameters. Assume to be sorted.
            params (numpy.ndarray | dict{str, numpy.ndarray}):
                Parameters for each time point in `t_params`.
            init_cond (numpy.ndarray | dict{str, numpy.ndarray}):
                Initial condition.

        Returns:
            soln (numpy.ndarray):
                Solutions for each time point in `t`.
        """
        assert len(params) == self.num_params
        assert len(init_cond) == self.num_components

        if isinstance(params, dict):
            params = np.vstack([
                params[name] for name in self.params
            ])

        if isinstance(init_cond, dict):
            init_cond = np.array([
                init_cond[name] for name in self.components
            ])

        assert hasattr(t, '__iter__')
        t = np.sort(np.unique(np.array(t)))
        assert t.size >= 2

        return self.solver.solve(t, t_params, params, init_cond)
