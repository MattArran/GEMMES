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

l_kl_min, l_kl_max, n_kl = 3., 8., 501
dl_kl = (l_kl_max - l_kl_min) / n_kl
geom_kl_ratios = 10**np.linspace(l_kl_min, l_kl_max, n_kl)
w0, f_prefix, n = 1e4, 100, 0.5
kl_sigma, kl_switch, switch_width = 0.1, 1e6, 2
                   
# ---------------------------
# user-defined function order (optional)

_FUNC_ORDER = None

_DESCRIPTION = """
    DESCRIPTION: Model with two classes of labor, with varying effectiveness on
        technologies with varying capital-labor ratio, fixed at creation time.
        For each class, wages converge to w0 L / (N - L) for w0 = $10k / yr,
        while class i productivity on a machine with capital-labor ratio k is
        f(k) = \epsilon_i(k) f0 (k/k0)^n, for 0-1 switching fns \epsilon_i
        and for f0 = $10k / human yr, k0 = $10k / human, n = 1/2
    TYPICAL BEHAVIOUR: ?
    LINKTOARTICLE: New
    """

# ---------------------------
# user-defined model
# contains parameters and functions of various types


def switch_functions(kl_ratios=np.arange(1, n_kl + 1)):
    return np.vstack([np.ones(kl_ratios.size), np.zeros(kl_ratios.size)])
    #switch = np.tanh(np.log(kl_ratios / kl_switch) / np.log(switch_width))
    #return np.vstack([0.5 * (1 + sign * switch) for sign in [-1, 1]])


def lognormal_distribution(kl_optimum=1, kl_sigma=1,
                           kl_ratios=np.arange(1, n_kl + 1)):
    gaussian = (np.exp(-0.5 * np.log(kl_ratios / kl_optimum)**2 / kl_sigma**2)
                / kl_ratios)
    weighted_sum = (gaussian[:-1] * np.diff(kl_ratios)).sum()
    return np.pad(gaussian[:-1], (0, 1)) / weighted_sum


def del_t_labor_density(I=0, I_distribution=np.ones(n_kl), delta=0,
                        kl_ratios=np.arange(1, n_kl + 1), itself=np.zeros(n_kl)):
    return I * I_distribution / kl_ratios - delta * itself


def is_labor_employed(productivity_fn=np.arange(n_kl),
                      labor_effectiveness=np.ones([2, n_kl]),
                      w=np.zeros(2)):
    w_array = np.atleast_2d(w)
    labor_employment = []
    for wages in w_array:
        profit = np.vstack([np.zeros(n_kl),
                            labor_effectiveness * productivity_fn - wages[:, np.newaxis]])
        labor_employment.append(profit[1:] == profit.max(axis=0))
    if w.ndim == 1:
        return labor_employment[0]
    return np.array(labor_employment)


def profit_maximising_kl_ratio(kl_ratios=np.arange(1, n_kl + 1),
                               productivity_fn=np.arange(n_kl),
                               labor_effectiveness=np.ones([2, n_kl]),
                               w=np.zeros(2)):
    w_array = np.atleast_2d(w)
    pfmkr = []
    for wages in w_array:
        rate_of_return = (labor_effectiveness * productivity_fn - wages[:, np.newaxis]) / kl_ratios
        pfmkr.append(kl_ratios[np.argmax(rate_of_return) % n_kl])
    if w.ndim == 1:
        return pfmkr[0]
    return np.array(pfmkr)


def total_production(kl_ratios=np.arange(1, n_kl + 1),
                     productivity_fn=np.arange(n_kl),
                     labor_effectiveness=np.ones([2, n_kl]),
                     labor_employed=np.ones([2, n_kl]),
                     labor_density=np.ones(n_kl)):    
    productivity = labor_employed * labor_effectiveness * productivity_fn
    production_density = (productivity * labor_density)[..., :-1]
    return (production_density * np.diff(kl_ratios)).sum(axis=(-1, -2))


def total_labor(kl_ratios=np.arange(1, n_kl + 1),
                labor_employed=np.ones([2, n_kl]),
                labor_density=np.ones(n_kl)):
    employed_labor_density = (labor_employed * labor_density)[..., :-1]
    return (employed_labor_density * np.diff(kl_ratios)).sum(axis=-1)

_PRESETS = {
    'default': {
        'fields': {
            'delta': 0.02,
            'beta': 0.,
            'tau': 0.5,
            's_p': 1,
            's_w': 0,
            'kl_sigma': kl_sigma,
            'kl_ratios': geom_kl_ratios,
            'N': 3e6 * np.ones(2),
            'w': np.array([50e3, 120e3]),
            'labor_density': (2.5e6 * lognormal_distribution(0.5e6, kl_sigma, geom_kl_ratios)
                              + 2.7e6 * lognormal_distribution(6.3e6, kl_sigma, geom_kl_ratios)),
        },
        'com': "System close to a Solow point, to which it rapidly converges",
        'plots': [],
    },
}

_DPARAM = {
    # ---------
    # exogenous parameters
    # can also have a time variation (just replace by a function of time)
    # useful for studying the model's reaction to an exogenous shock
    'Tmax' : 100,
    'dt': 0.01,
    'delta': 0.02,
    'beta': 0.,
    'tau': 1.0,
    's_p': 1,
    's_w': 0,
    'phinull': 1./6.,
    'kl_sigma': kl_sigma,
    'kl_ratios': geom_kl_ratios,

    # ---------
    # endogeneous functions

    # differential equations (pde)
    'N': {
        'func': lambda beta=0, itself=0: beta * itself,
        'initial': np.array([3e6, 3e6]),
        'eqtype': 'pde',
    },
    'w': {
        'func': lambda itself=0, phillips=0: itself * phillips,
        'initial': np.array([5e3, 12e3]),
        'eqtype': 'pde',
    },
    'labor_density': {
        'func': del_t_labor_density,
        'initial': 2.5e5 * lognormal_distribution(0.5e6, kl_sigma, geom_kl_ratios),
                   #(2.5e6 * lognormal_distribution(0.5e6, kl_sigma, geom_kl_ratios)
                   # + 2.7e6 * lognormal_distribution(6.3e6, kl_sigma, geom_kl_ratios)),
        'eqtype': 'pde',
    },

    # Intermediary functions (endogenous, computation intermediates)
    'productivity_fn': {
        'func': lambda kl_ratios=np.arange(n_kl): f_prefix * kl_ratios**n,
        'eqtype': 'intermediary',
    },
    'labor_effectiveness': {
        'func': switch_functions,
        'eqtype': 'intermediary',
    },
    'labor_employed': {
        'func': is_labor_employed,
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
        'func': lambda GDP=0, w=0, L=0: GDP - np.array(w * L).sum(axis=-1),
        'eqtype': 'intermediary',
    },
    'I': {
        'func': lambda s_p=1, s_w=0, Pi=0, w=0, L=0: s_p * Pi + s_w * np.array(w * L).sum(axis=-1),
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
