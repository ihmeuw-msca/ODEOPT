# -*- coding: utf-8 -*-
"""
    test ode solver
    ~~~~~~~~~~~~~~~
"""
import numpy as np
import pytest
from odeopt.ode import ForwardEuler, RK4


@pytest.fixture()
def simple_system():
    def system(t, y, lam):
        return lam[0] * y

    return system


@pytest.mark.parametrize('dt', [1e-1, 1e-2, 1e-3])
@pytest.mark.parametrize('t_span', [np.array([0.0, 1.0])])
@pytest.mark.parametrize(('params', 'init_cond', 'true_fun'),
                         [(np.array([[-1.0]]),
                           np.array([1.0]),
                           lambda t: np.exp(-t))])
def test_forward_euler(simple_system, dt, t_span, init_cond, params, true_fun):
    solver = ForwardEuler(simple_system, dt)
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    result = true_fun(t)
    my_result = solver.solve(t, init_cond,
                             t, np.repeat(params, t.size, axis=1))[0]
    assert np.sqrt(np.mean((result - my_result) ** 2)) < dt * 5.0


@pytest.mark.parametrize('dt', [1e-1, 1e-2, 1e-3])
@pytest.mark.parametrize('t_span', [np.array([0.0, 1.0])])
@pytest.mark.parametrize(('params', 'init_cond', 'true_fun'),
                         [(np.array([[-1.0]]),
                           np.array([1.0]),
                           lambda t: np.exp(-t))])
def test_runge_kutta_4th(simple_system, dt, t_span, init_cond, params, true_fun):
    solver = RK4(simple_system, dt)
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    result = true_fun(t)
    my_result = solver.solve(t, init_cond,
                             t, np.repeat(params, t.size, axis=1))[0]
    assert np.sqrt(np.mean((result - my_result) ** 2)) < dt ** 4 * 5.0
