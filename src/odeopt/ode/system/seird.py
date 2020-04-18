# -*- coding: utf-8 -*-
"""
    SEIRD Model
    ~~~~~~~~~~~
"""
from .odesys import ODESys


class SEIRD(ODESys):
    """SEIRD Model.
    """
    def __init__(self):
        """Constructor function for SEIR.
        """
        # create system
        def system(t, y, p):
            aI = p['alphaI'](t)
            aS = p['alphaS'](t)
            aN = p['alphaN'](t)
            b = p['beta'](t)
            n = p['N'](t)
            r = p['gamma'](t)
            s = p['sigma'](t)
            x = p['chi'](t)

            ds = -b*(y[0]**aS)*(y[2]**aI)/(n**aN)
            de = b*(y[0]**aS)*(y[2]**aI)/(n**aN) - s*y[1]
            di = s*y[1] - (r + x)*y[2]
            dr = r*y[2]
            dd = x*y[2]

            return [ds, de, di, dr, dd]

        # create parameters
        params = {
            'alphaI': lambda x: 1.0,
            'alphaS': lambda x: 1.0,
            'alphaN': lambda x: 1.0,
            'beta': lambda x: 1.0,
            'N': lambda x: 1.0,
            'gamma': lambda x: 0.5,
            'sigma': lambda x: 0.8,
            'chi': lambda x: 0.01,
        }

        # create component names
        component_names = ['S', 'E', 'I', 'R', 'D']

        super().__init__(system, params, component_names)
