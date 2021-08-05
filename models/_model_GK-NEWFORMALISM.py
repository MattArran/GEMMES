# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:11:15 2021

@author: Paul Valcke
"""

import numpy as np


# ---------------------------
_DESCRIPTION = """

    DESCRIPTION : This is a Goodwin model based on extensive variables.
    Inflation not integrated to the process
    TYPICAL BEHAVIOR : Convergence toward solow point ( good equilibrium) or debt crisis
    LINKTOARTICLE :
    """   # Description that will be shown when the user load the model or browse through models


# ---------------------------
# user-defined model
# contains the logical core
_LOGICS = {
    # differential equations (ode)
    'ode': {
        'a': {
            'logic': lambda alpha=0, itself=0: alpha * itself,
            'com': 'Exogenous technical progress as an exponential',
        },
        'N': {
            'logic': lambda beta=0, itself=0: beta * itself,
            'com': 'Exogenous population as an exponential',
        },
        'K': {
            'logic': lambda I=0, itself=0, delta=0: I - itself * delta,
            'com': 'Capital evolution from investment and depreciation',
        },
        'D': {
            'logic': lambda I=0, Pi=0: I - Pi,
            'com': 'Debt as Investment-Profit difference',
        },
        'W': {
            'logic': lambda phillips=0, itself=0: itself*phillips,
            'com': 'salary through negociation',
        },
        'p': {
            'logic': lambda itself=0, inflation=0: itself*inflation,
            'com': 'NO INFLATION FOR THE MOMENT',
            'initial': 1,
        },
    },

    # Intermediary relevant functions
    'statevar': {
        'Y': {
            'logic': lambda K=0, nu=1: K / nu,
            'com': 'Leontiev optimized production function ',
        },
        'GDP': {
            'logic': lambda Y=0, p=0: Y*p,
            'com': 'Output with selling price ',
        },
        'inflation': {
            'logic': lambda p=0: 0,
            'com': 'INFLATION NOT CODED',
        },
        'L': {
            'logic': lambda K=0, a=1, nu=1: K / (a * nu),
            'com': 'Full instant employement based on capital',
        },
        'Pi': {
            'logic': lambda Y=0, W=0, L=0, r=0, D=0: Y - W * L - r * D,
            'com': 'Profit for production-Salary-debt logic',
        },
        'lambda': {
            'logic': lambda L=0, N=1: L / N,
            'com': 'employement rate',
        },
        'omega': {
            'logic': lambda W=0, L=0, Y=1: W * L / Y,
            'com': 'wage share',
        },
        'phillips': {
            'logic': lambda phi0=0, phi1=0, lamb=0: -phi0 + phi1 / (1 - lamb)**2,
            'com': 'Wage increase rate through employement',
        },
        'kappa': {
            'logic': lambda k0=0, k1=0, k2=0, Pi=0, Y=1: k0 + k1 * np.exp(k2*Pi/Y),
            'com': 'Relative GDP investment through relative profit',
        },
        'I': {
            'logic': lambda GDP=0, kappa=0: GDP * kappa,
            'com': 'Investment value',
        },
        'd': {
            'logic': lambda D=0, GDP=1: D / GDP,
            'com': 'private debt ratio',
        },
        'pi': {
            'logic': lambda omega=0, r=0, d=0: 1 - omega - r * d,
            'com': 'relative profit',
        },
    },
    'parameters': {
        'nu': {
            'logic': 3,
            'com': 'OVERLOADING A DEFINITION',
        }
    }
}


_PRESETS = {
    'default': {
        'fields': {
            'a': 1,
            'N': 1,
            'K': 2.7,
            'D': 0.2,
            'W': .85,
            'p': 1,
            'alpha': 0.02,
            'beta': 0.025,
            'nu': 3,
            'delta': .005,
            'phinull': .04,
            'k0': -0.0065,
            'k1': np.exp(-5),
            'k2': 20,
            'r': 0.03,
        },
        'com': (
            'This is a run that should give simple '
            'convergent oscillations'),
        'plots': [],
    },
    'crisis': {
        'fields': {
            'a': 1,
            'N': 1,
            'K': 2.7,
            'D': 10,
            'W': .85,
            'p': 1,

            'r': 0.0
        },
        'com': (
            'This is a run that should create a debt crisis'
        ),
        'plots': [],
    },
}
