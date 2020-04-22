# -*- coding: utf-8 -*-
"""
    Linear ODE System
    ~~~~~~~~~~~~~~~~~
"""
import numpy as np
from .odesys import ODESys


class LinearSIR(ODESys):
    """Linear SIR Model.
    """
    def __init__(self, *args):
        """Constructor of BetaSEIR Model.
        """
        # create the system
        def system(t, y, p):
            beta = p[0]
            gamma = p[1]

            s = y[0]
            i = y[1]
            r = y[2]

            ds = -beta*i
            di = (beta - gamma)*i
            dr = gamma*i

            return np.array([ds, di, dr])

        # create parameter names
        params = ['beta', 'gamma']

        # create component names
        components = ['S', 'I', 'R']

        super().__init__(system, params, components, *args)


class LinearSEIR(ODESys):
    """Linear SEIR Model.
    """
    def __init__(self, *args):
        """Constructor of BetaSEIR Model.
        """
        # create the system
        def system(t, y, p):
            beta = p[0]
            sigma = p[1]
            gamma = p[2]

            s = y[0]
            e = y[1]
            i = y[2]
            r = y[3]

            ds = -beta*i
            de = beta*i - sigma*e
            di = sigma*e - gamma*i
            dr = gamma*i

            return np.array([ds, de, di, dr])

        # create parameter names
        params = ['beta', 'sigma', 'gamma']

        # create component names
        components = ['S', 'E', 'I', 'R']

        super().__init__(system, params, components, *args)
