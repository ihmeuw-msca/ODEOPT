# -*- coding: utf-8 -*-
"""
    ODE Solver
    ~~~~~~~~~~

    Parent class of the ODE Solver.
"""
import numpy as np
from odeopt.core import utils


class ODESolver:
    """ODE Solver.
    """
    def __init__(self, system, dt):
        """Constructor of the ODESovler.

        Args:
            system (callable):
                System of ODE
            dt (float):
                Step size for solving the ODE.
        """
        self.system = system
        self.dt = dt

    def solve(self, t, t_params, params, init_cond):
        """Solve the ODE.

        Args:
            t (numpy.ndarray):
                Time points where we evaluate the system of ODE. Assume to be
                sorted.
            t_params (numpy.ndarray):
                Time stamp for the parameters. Assume to be sorted.
            params (numpy.ndarray):
                Parameters for each time point in `t_params`.
            init_cond (numpy.ndarray):
                Initial condition.

        Returns:
            soln (numpy.ndarray):
                Solutions for each time point in `t`.
        """
        pass


class ForwardEuler(ODESolver):
    """Forward Euler Solver.
    """
    def __init__(self, *args):
        super().__init__(*args)

    def solve(self, t, t_params, params, init_cond):
        """Forward Euler solver.
        """
        t_solve = np.arange(np.min(t), np.max(t) + self.dt, self.dt)
        y_solve = np.zeros((init_cond.size, t_solve.size),
                           dtype=init_cond.dtype)
        y_solve[:, 0] = init_cond
        # linear interpolate the parameters
        params = utils.linear_interpolate(t_solve, t_params, params)
        for i in range(1, t_solve.size):
            y_solve[:, i] = y_solve[:, i - 1] + self.dt*self.system(
                t_solve[i], y_solve[:, i - 1], params[:, i - 1])

        # linear interpolate the solutions.
        y_solve = utils.linear_interpolate(t, t_solve, y_solve)
        return y_solve
