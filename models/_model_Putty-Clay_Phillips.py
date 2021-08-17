# -*- coding: utf-8 -*-
"""
Here we define the parameters and set of equation for a model of type 'Putty-Clay'

All parameters can have value:
    - None: set to the default value found in _def_fields.py
    - scalar: int or float
    - list or np.ndarray used for benchmarks
    - function (callable): in that can it will be treated as a variable
                            the function will be called at each time step

"""


import numpy as np

n_kl = 501
geom_kl_ratios = 10**np.linspace(3, 8, n_kl)
w0, f_prefix, n = 1e4, 100, 0.5
                   
# ---------------------------
# user-defined function order (optional)

_FUNC_ORDER = None

_DESCRIPTION = """
    DESCRIPTION: Model with wages uniquely determined by unemployment, but with
        technologies with varying capital-labor ratio, fixed at creation time.
        In this version, wages converge to w0 L / (N - L) for w0 = $10k / yr,
        while labor productivity on a machine with capital-labor ratio k is
        f(k) = f0 (k/k0)^n for f0 = $10k / human yr, k0 = $10k / human, n = 1/2
    TYPICAL BEHAVIOUR: Decaying oscillations around a Solow point (?)
    LINKTOARTICLE: Akerlof, G. A. and Stiglitz, J. E., 1969. 'Capital, Wages
        and Structural Unemployment', The Economic Journal, Vol. 79, No. 314
        http://www.jstor.org/stable/2230168
    """

_PRESETS = {
    'default': {
        'fields': {
            'delta': 0.05,
            'beta': 0.,
            's_p': 1,
            's_w': 0,
            'kl_sigma': 0.1,
            'kl_ratios': geom_kl_ratios,
            'phinull': 1./6.,
            'N': 6e6,
            'w': 5e4,
            'labor_density': (5 / (10**0.05 - 1)
                              * np.where(np.abs(np.log10(geom_kl_ratios / 1e6)) < 0.025, 1, 0)),
        },
        'com': "System close to a Solow point, to which it slowly converges",
        'plots': [],
    },
    
    'development': {
        'fields': {
            'delta': 0.05,
            'beta': 0.,
            's_p': 1,
            's_w': 0,
            'kl_sigma': 0.1,
            'kl_ratios': geom_kl_ratios,
            'phinull': 1./6.,
            'N': 6e6,
            'w': 5e3,
            'labor_density': 1e6 * np.exp(-0.5 * (np.log(geom_kl_ratios / 1e4) 
                                                  / np.log(2))**2) / geom_kl_ratios,
        },
        'com': "System develops from a labor-intensive, low-wage start point",
        'plots': [],
    }
}

# ---------------------------
# user-defined model
# contains parameters and functions of various types


def del_t_labor_density(I=0, I_distribution=np.ones(n_kl), delta=0,
                        kl_ratios=np.arange(1, n_kl + 1), itself=np.zeros(n_kl)):
    return I * I_distribution / kl_ratios - delta * itself


def minimum_profitable_kl_ratio_index(productivity_fn=np.arange(n_kl), w=0):
    return np.nonzero(productivity_fn >= w)[0][0]


def profit_maximising_kl_ratio(productivity_fn=np.arange(n_kl), w=0,
                               kl_ratios=np.arange(1, n_kl + 1)):
    return kl_ratios[np.argmax((productivity_fn - w) / kl_ratios)]


def lognormal_distribution(kl_optimum=1, kl_sigma=1,
                           kl_ratios=np.arange(1, n_kl + 1)):
    gaussian = (np.exp(-0.5 * np.log(kl_ratios / kl_optimum)**2 / kl_sigma**2)
                / kl_ratios)
    weighted_sum = (gaussian[:-1] * np.diff(kl_ratios)).sum()
    return np.pad(gaussian[:-1], (0, 1)) / weighted_sum


def total_production(id_kl_min=0, productivity_fn=np.arange(n_kl),
                     labor_density=np.ones(n_kl),
                     kl_ratios=np.arange(1, n_kl + 1)):
    if np.isscalar(id_kl_min):
        return (productivity_fn[id_kl_min:-1] * labor_density[id_kl_min:-1]
                * np.diff(kl_ratios[id_kl_min:])).sum()
    ltp = [(productivity_fn[i:-1] * ld[i:-1] * np.diff(kl_ratios[i:])).sum()
           for i, ld in zip(id_kl_min.astype(int), labor_density)]
    return np.array(ltp)


def total_labor(id_kl_min=0, labor_density=np.ones(n_kl),
                kl_ratios=np.arange(1, n_kl + 1)):
    if np.isscalar(id_kl_min):
        return (labor_density[id_kl_min:-1] * np.diff(kl_ratios)).sum()
    ltl = [(ld[i:-1] * np.diff(kl_ratios[i:])).sum()
           for i, ld in zip(id_kl_min.astype(int), labor_density)]
    return np.array(ltl)


_DPARAM = {
    # ---------
    # exogenous parameters
    # can also have a time variation (just replace by a function of time)
    # useful for studying the model's reaction to an exogenous shock
    'Tmax' : 500,
    'delta': 0.05,
    'beta': 0.,
    's_p': 1,
    's_w': 0,
    'phinull': 1./6.,
    'kl_sigma': 0.1,
    'kl_ratios': geom_kl_ratios,

    # ---------
    # endogeneous functions

    # differential equations (ode)
    'N': {
        'func': lambda beta=0, itself=0: beta * itself,
        'initial': 6e6,
        'eqtype': 'ode',
    },
    'w': {
        'func': lambda itself=0, phillips=0: itself * phillips,
        'initial': 5e3,
        'eqtype': 'ode',
    },
    # differential equations (pde)
    'labor_density': {
        'func': del_t_labor_density,
        'initial': 1e6 * np.exp(-0.5 * (np.log(geom_kl_ratios / 1e4) 
                                        / np.log(2))**2) / geom_kl_ratios,
        'eqtype': 'pde',
    },

    # Intermediary functions (endogenous, computation intermediates)
    'productivity_fn': {
        'func': lambda kl_ratios=np.arange(n_kl): f_prefix * kl_ratios**n,
        'eqtype': 'intermediary',
    },
    'id_kl_min': {
        'func': minimum_profitable_kl_ratio_index,
        'eqtype': 'intermediary',
    },
    'kl_optimum': {
        'func': profit_maximising_kl_ratio,
        'eqtype': 'intermediary',
    },
    'I_distribution': {
        'func': lognormal_distribution,
        'eqtype': 'intermediary',
    },
    'GDP': {
        'func': total_production,
        'eqtype': 'intermediary',
    },
    'L': {
        'func': total_labor,
        'eqtype': 'intermediary',
    },
    'Pi': {
        'func': lambda GDP=0, w=0, L=0: GDP - w * L,
        'eqtype': 'intermediary',
    },
    'I': {
        'func': lambda s_p=1, s_w=0, Pi=0, w=0, L=0: s_p * Pi + s_w * w * L,
        'eqtype': 'intermediary',
    },
    'phillips': {
        'func': lambda phi0=0, phi1=0, L=0, N=1: -phi0 + phi1 / (1 - L / N)**2,
        'eqtype': 'intermediary',
    },
    'phi0': {
        'func': lambda phinull=0: phinull / (1 - phinull**2),
        'eqtype': 'auxiliary',
    },
    'phi1': {
        'func': lambda phinull=0: phinull**3 / (1 - phinull**2),
        'eqtype': 'auxiliary',
    },

    # auxiliary, not used for computation but for interpretation
    # => typically computed at the end after the computation
    'lambda': {
        'func': lambda L=0, N=1: L / N,
        'eqtype': 'auxiliary',
    },
    'omega': {
        'func': lambda w=0, L=0, GDP=1: w * L / GDP,
        'eqtype': 'auxiliary',
    }
}
