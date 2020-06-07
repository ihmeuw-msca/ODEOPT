# -*- coding: utf-8 -*-
"""
    Nonlinear ODE System
    ~~~~~~~~~~~~~~~~~~~~
"""
from typing import Callable, Union
import numpy as np
from .odesys import ODESys


class SEIRD(ODESys):
    """SEIRD Model.
    """
    def __init__(self, *args):
        """Constructor function for SEIR.
        """
        # create system
        def system(t, y, p):
            aI = p[0]
            aS = p[1]
            aN = p[2]
            b = p[3]
            n = p[4]
            s = p[5]
            r = p[6]
            x = p[7]

            ds = -b*(y[0]**aS)*(y[2]**aI)/(n**aN)
            de = b*(y[0]**aS)*(y[2]**aI)/(n**aN) - s*y[1]
            di = s*y[1] - (r + x)*y[2]
            dr = r*y[2]
            dd = x*y[2]

            return np.array([ds, de, di, dr, dd])

        # create parameters
        params = [
            'alphaI', 'alphaS', 'alphaN',
            'beta', 'N', 'sigma',
            'gamma', 'chi',
        ]

        # create component names
        components = ['S', 'E', 'I', 'R', 'D']

        super().__init__(system, params, components, *args)


class SimpleSEIRD(ODESys):
    """SEIRD Model.
    """
    def __init__(self, *args):
        """Constructor function for SimpleSEIR.
        """
        # create system
        def system(t, y, p):
            b = p[0]
            s = p[1]
            r = p[2]
            x = p[3]

            ds = -b*y[0]*y[2]
            de = b*y[0]*y[2] - s*y[1]
            di = s*y[1] - (r + x)*y[2]
            dr = r*y[2]
            dd = x*y[2]

            return np.array([ds, de, di, dr, dd])

        # create parameter names
        params = [ 'beta', 'sigma', 'gamma', 'chi']

        # create component names
        components = ['S', 'E', 'I', 'R', 'D']

        super().__init__(system, params, components, *args)


class BetaSEIR(ODESys):
    """SEIR Model that only have beta as parameter.
    """
    def __init__(self, alpha, sigma, gamma, N, *args):
        """Constructor of BetaSEIR Model.
        """
        # create system
        assert 0.0 < alpha <= 1.0
        assert sigma >= 0.0
        assert gamma >= 0.0
        assert N > 0.0
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

        # create parameter names
        params = ['beta']

        # create component names
        components = ['S', 'E', 'I', 'R']

        super().__init__(self.system, params, components, *args)

    def update_given_params(self, alpha=None, sigma=None, gamma=None, N=None,):
        """Update given parameters.

        Args:
            alpha (float | None, optional):
                Updated alpha parameter, if `None` no update will happen.
            sigma (float | None, optional):
                Updated sigma parameter, if `None` no update will happen.
            gamma (float | None, optional):
                Updated gamma parameter, if `None` no update will happen.
            N (float | None, optional):
                Update N parameter, if `None` no update will happen.
        """
        if alpha is not None:
            assert 0.0 < alpha <= 1.0
            self.alpha = alpha
        if sigma is not None:
            assert sigma >= 0.0
            self.sigma = sigma
        if gamma is not None:
            assert gamma >= 0.0
            self.gamma = gamma
        if N is not None:
            assert N > 0.0
            self.N = N

    def system(self, t, y, p):
        beta = p[0]

        s = y[0]
        e = y[1]
        i = y[2]
        r = y[3]

        ds = -beta*(s/self.N)*i**self.alpha
        de = beta*(s/self.N)*i**self.alpha - self.sigma*e
        di = self.sigma*e - self.gamma*i
        dr = self.gamma*i

        return np.array([ds, de, di, dr])


class BetaSEIIR(ODESys):
    """SEIIR Model that only have beta as parameter.
    """
    def __init__(self, alpha, sigma, gamma1, gamma2, N, *args):
        """Constructor of BetaSEIR Model.
        """
        # create system
        assert 0 < alpha <= 1.0
        assert sigma >= 0.0
        assert gamma1 >= 0.0
        assert gamma2 >= 0.0
        assert N > 0.0
        self.alpha = alpha
        self.sigma = sigma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.N = N

        # create parameter names
        params = ['beta']

        # create component names
        components = ['S', 'E', 'I1', 'I2', 'R']

        super().__init__(self.system, params, components, *args)

    def update_given_params(self,
                            alpha=None,
                            sigma=None,
                            gamma1=None,
                            gamma2=None,
                            N=None):
        """Update given parameters.

        Args:
            alpha (float | None, optional):
                Updated alpha parameter, if `None` no update will happen.
            sigma (float | None, optional):
                Updated sigma parameter, if `None` no update will happen.
            gamma1 (float | None, optional):
                Updated gamma1 parameter, if `None` no update will happen.
            gamma2 (float | None, optional):
                Updated gamma2 parameter, if `None` no update will happen.
            N (float | None, optional):
                Update N parameter, if `None` no update will happen.
        """
        if alpha is not None:
            assert 0.0 < alpha <= 1.0
            self.alpha = alpha
        if sigma is not None:
            assert sigma >= 0.0
            self.sigma = sigma
        if gamma1 is not None:
            assert gamma1 >= 0.0
            self.gamma1 = gamma1
        if gamma2 is not None:
            assert gamma2 >= 0.0
            self.gamma2 = gamma2
        if N is not None:
            assert N > 0.0
            self.N = N

    def system(self, t, y, p):
        beta = p[0]

        s = y[0]
        e = y[1]
        i1 = y[2]
        i2 = y[3]
        r = y[4]

        ds = -beta*(s/self.N)*(i1 + i2)**self.alpha
        de = beta*(s/self.N)*(i1 + i2)**self.alpha - self.sigma*e
        di1 = self.sigma*e - self.gamma1*i1
        di2 = self.gamma1*i1 - self.gamma2*i2
        dr = self.gamma2*i2

        return np.array([ds, de, di1, di2, dr])


class BetaSEIIRWithFun(BetaSEIIR):
    """SEIIR Model with function wrap around (I1 + I2).
    """
    def __init__(self, fun: Callable, *args):
        """Constructor for BetaSEIIRWithFun.

        Args:
            fun (Callable): Function that modifies I1 + I2.
        """
        assert callable(fun), "fun has to be callable."
        self.fun = fun
        super().__init__(*args)

    def update_given_params(self, fun: Union[Callable, None] = None, **kwargs):
        """Update given parameters.
        """
        if fun is not None:
            assert callable(fun)
            self.fun = fun
        super().update_given_params(**kwargs)

    def system(self, t, y, p):
        beta = p[0]

        s = y[0]
        e = y[1]
        i1 = y[2]
        i2 = y[3]
        r = y[4]

        ds = -beta*(s/self.N)*self.fun(i1 + i2)**self.alpha
        de = beta*(s/self.N)*self.fun(i1 + i2)**self.alpha - self.sigma*e
        di1 = self.sigma*e - self.gamma1*i1
        di2 = self.gamma1*i1 - self.gamma2*i2
        dr = self.gamma2*i2

        return np.array([ds, de, di1, di2, dr])


class SEIR(ODESys):
    """SEIR Model.
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

            ds = -beta*s*i
            de = beta*s*i - sigma*e
            di = sigma*e - gamma*i
            dr = gamma*i

            return np.array([ds, de, di, dr])

        # create parameter names
        params = ['beta', 'sigma', 'gamma']

        # create component names
        components = ['S', 'E', 'I', 'R']

        super().__init__(system, params, components, *args)


class SIR(ODESys):
    """SIR Model.
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

            ds = -beta*s*i
            di = beta*s*i - gamma*i
            dr = gamma*i

            return np.array([ds, di, dr])

        # create parameter names
        params = ['beta', 'gamma']

        # create component names
        components = ['S', 'I', 'R']

        super().__init__(system, params, components, *args)
