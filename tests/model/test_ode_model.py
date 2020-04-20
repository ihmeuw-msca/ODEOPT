# -*- coding: utf-8 -*-
"""
    test_ode_model
    ~~~~~~~~~~~~~~
"""
import numpy as np
import pandas as pd
from odeopt.core import ODEData
from odeopt.ode import ODESys
from odeopt.model import SingleInitModel, InitModel
from odeopt.model import SingleParamModel, ParamModel
from odeopt.model import ODEModel


# create true generating mechanism
# assume we have three groups,
# and we consider a simple ode
lam_A = -1.0
lam_B = -2.0
lam_C = -3.0

def true_soln(t, lam=-1.0):
    return np.exp(lam*t)

times = np.linspace(0.0, 2.0, 5)

y_A = true_soln(times, lam=lam_A)
y_B = true_soln(times, lam=lam_B)
y_C = true_soln(times, lam=lam_C)

# create the data frame
test_df = pd.DataFrame({
    'group': ['A']*5 + ['B']*5 + ['C']*5,
    't': np.hstack([times, times, times]),
    'y': np.hstack([y_A, y_B, y_C])
})
test_data = ODEData(test_df, 'group', 't', ['y'])


# create the ode system
test_ode_sys = ODESys(lambda t, y, params: params[0]*y,
                      ['lam'], ['y'],
                      solver_dt=1e-2)

# create the init model
test_init_model = InitModel([
    SingleInitModel('y',
                    link_fun=np.exp,
                    use_re=False,
                    fe_bounds=[-1.0, 1.0])
])

# create the parameter model
test_param_model = ParamModel([
    SingleParamModel('lam', ['intercept'],
                     use_re=True,
                     fe_bounds=[-np.inf, 0.0],
                     re_bounds=[-np.inf, 0.0])
])

# create the ode model
ode_model = ODEModel(test_data,
                     test_ode_sys,
                     test_init_model,
                     test_param_model)

def test_ode_model():
    assert ode_model.num_groups == 3
    assert ode_model.init_var_size == 1
    assert ode_model.param_var_size == 4
    assert ode_model.optvar_bounds.shape == (5, 2)


# initialization
x0 = np.array([0.0, -1.0, 0.0, -1.0, -2.0])
ode_model.fit_model(x0)

def test_ode_model_fit_model():
    assert np.abs(ode_model.result_params['A'][0][0] - lam_A) < 0.1
    assert np.abs(ode_model.result_params['B'][0][0] - lam_B) < 0.1
    assert np.abs(ode_model.result_params['C'][0][0] - lam_C) < 0.1
