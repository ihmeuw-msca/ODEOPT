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
    def __init__(self, **kwargs):
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

        super().__init__(system, params, components, **kwargs)


class LinearSEIR(ODESys):
    """Linear SEIR Model.
    """
    def __init__(self, **kwargs):
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

        super().__init__(system, params, components, **kwargs)


class LinearBetaSEIR(ODESys):
    """SEIR Model that only have beta as parameter.
    """
    def __init__(self, sigma, gamma, **kwargs):
        """Constructor of BetaSEIR Model.
        """
        # create system
        assert sigma >= 0.0
        assert gamma >= 0.0
        self.sigma = sigma
        self.gamma = gamma

        # create parameter names
        params = ['beta']

        # create component names
        components = ['S', 'E', 'I', 'R']

        super().__init__(self.system, params, components, **kwargs)

    def update_given_params(self, sigma=None, gamma=None):
        """Update given parameters.

        Args:
            sigma (float | None, optional):
                Updated sigma parameter, if `None` no update will happen.
            gamma (float | None, optional):
                Updated gamma parameter, if `None` no update will happen.
        """
        if sigma is not None:
            assert sigma >= 0.0
            self.sigma = sigma
        if gamma is not None:
            assert gamma >= 0.0
            self.gamma = gamma

    def system(self, t, y, p):
        beta = p[0]

        s = y[0]
        e = y[1]
        i = y[2]
        r = y[3]

        ds = -beta*i
        de = beta*i - self.sigma*e
        di = self.sigma*e - self.gamma*i
        dr = self.gamma*i

        return np.array([ds, de, di, dr])


class LinearBetaSIR(ODESys):
    """Linear BetaSIR Model.
    """
    def __init__(self, gamma, **kwargs):
        """Constructor of BetaSEIR Model.
        """
        # create the system
        assert gamma >= 0.0
        self.gamma = gamma

        # create parameter names
        params = ['beta']

        # create component names
        components = ['S', 'I', 'R']

        super().__init__(self.system, params, components, **kwargs)

    def update_given_params(self, gamma=None):
        """Update given parameters.

        Args:
            gamma (float | None, optional):
                Updated gamma parameter, if `None` no update will happen.
        """
        if gamma is not None:
            assert gamma >= 0.0
            self.gamma = gamma

    def system(self, t, y, p):
        beta = p[0]

        s = y[0]
        i = y[1]
        r = y[2]

        ds = -beta*i
        di = (beta - self.gamma)*i
        dr = self.gamma*i

        return np.array([ds, di, dr])


class LinearFirstOrder(ODESys):
    """Linear First Order ODE.
    """
    def __init__(self, c, **kwargs):
        """Constructor of LinearFirstOrder Model.
        """
        # create the system
        assert np.isscalar(c)
        self.c = c

        # create parameter names
        params = ['f']

        # create component names
        components = ['x']

        super().__init__(self.system, params, components, **kwargs)

    def update_given_params(self, c=None):
        """Update given parameters.

        Args:
            c (float | None, optional):
                Updated gamma parameter, if `None` no update will happen.
        """
        if c is not None:
            assert np.isscalar(c)
            self.c = c

    def system(self, t, y, p):
        f = p[0]
        x = y[0]
        dx = -self.c*x + f
        return np.array([dx])
