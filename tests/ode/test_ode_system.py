# -*- coding: utf-8 -*-
"""
    test ode system
    ~~~~~~~~~~~~~~~
"""
import numpy as np
import pytest
from odeopt.ode import ODESys
from odeopt.ode import SEIRD
from odeopt.ode import BetaSEIR
from odeopt.ode import ODESolver
from odeopt.ode import ForwardEuler


@pytest.fixture()
def simple_system():
    def system(t, y, lam):
        return lam[0]*y

    return ODESys(system, ['lam'], ['y'])


def test_odesys(simple_system):
    assert simple_system.num_params == 1
    assert simple_system.num_components == 1
    assert isinstance(simple_system.solver, simple_system.solver_class)
    assert simple_system.solver.dt == simple_system.solver_dt


@pytest.mark.parametrize('solver_class', [ODESolver, ForwardEuler])
@pytest.mark.parametrize('solver_dt', [0.1, 0.2])
def test_odesys_update_solver(simple_system, solver_class, solver_dt):
    simple_system.update_solver(solver_class=solver_class,
                                solver_dt=solver_dt)

    assert isinstance(simple_system.solver, solver_class)
    assert simple_system.solver.dt == solver_dt


def test_seird():
    seird = SEIRD()
    assert seird.num_params == 8
    assert seird.num_components == 5


@pytest.mark.parametrize(('sigma', 'new_sigma'), [(0.1, 0.2)])
@pytest.mark.parametrize(('gamma', 'new_gamma'), [(0.1, 0.2)])
@pytest.mark.parametrize('t', [0.0])
@pytest.mark.parametrize('y',
                         [[0.1, 0.1, 0.1, 0.1],
                          [0.2, 0.2, 0.2, 0.2]])
@pytest.mark.parametrize('p', [[0.1]])
def test_betaseir(sigma, new_sigma, gamma, new_gamma, t, y, p):
    ode_sys = BetaSEIR(sigma, gamma)
    result_1 = ode_sys.system(t, y, p)

    assert ode_sys.sigma == sigma
    assert ode_sys.gamma == gamma

    ode_sys.update_given_params(sigma=new_sigma,
                                gamma=new_gamma)
    result_2 = ode_sys.system(t, y, p)

    assert ode_sys.sigma == new_sigma
    assert ode_sys.gamma == new_gamma

    assert not np.allclose(result_1, result_2)
