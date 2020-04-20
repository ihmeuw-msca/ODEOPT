# -*- coding: utf-8 -*-
"""
    SEIRD Model
    ~~~~~~~~~~~
"""
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

        # create parameters
        params = [ 'beta', 'sigma', 'gamma', 'chi']

        # create component names
        components = ['S', 'E', 'I', 'R', 'D']

        super().__init__(system, params, components, *args)
