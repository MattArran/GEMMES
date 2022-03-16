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

l_kl_min, l_kl_max, n_kl = 3., 8., 1001
dl_kl = (l_kl_max - l_kl_min) / n_kl
geom_kl_ratios = 10**np.linspace(l_kl_min, l_kl_max, n_kl)
w0, f_prefix, n = 1e4, 100, 0.5
kl_sigma = 0.1

# ---------------------------
# user-defined function order (optional)

_FUNC_ORDER = None

_DESCRIPTION = """
    DESCRIPTION: Model with wages uniquely determined by unemployment, 
        investment a positive fraction of GDP, increasing with profit, and
        technologies with varying capital-labor ratio, fixed at creation time.
        In this version, wages converge to w0 L / (N - L) for w0 = $10k / yr,
        while labor productivity on a machine with capital-labor ratio k is
        f(k) = f0 (k/k0)^n for f0 = $10k / human yr, k0 = $10k / human, n = 1/2
    TYPICAL BEHAVIOUR: Decaying oscillations around a Solow point, or collapse
    LINKTOARTICLE: Akerlof, G. A. and Stiglitz, J. E., 1969. 'Capital, Wages
        and Structural Unemployment', The Economic Journal, Vol. 79, No. 314
        http://www.jstor.org/stable/2230168
    """

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

"""
At the Solow point, writing kl_opt for kl_optimum & f for productivity_fn:
           0 = \dot{D} = I - Pi, so ...
    Pi / GDP = I / GDP = k0 + k1 exp(k2 Pi / GDP), giving Pi / GDP & I / GDP.
           0 = \dot{L} = I / kl_opt - delta L, so ...
     L / GDP = (Pi / GDP) / (delta * kl_opt), giving L / GDP.
         GDP = f(kl_opt) L, so ...
    kl_opt / f(kl_opt) = (Pi / GDP) / delta, giving kl_opt.
           w = f(kl_opt) - kl_opt f'(kl_opt), giving w.
          Pi = Y - w L - r D, so ...
     D / GDP = (1 - w / f(kl_opt) - Pi / GDP) / r, giving D / GDP.
           w = w0 L / (N - L), giving L, hence GDP, Pi, I, D.
"""

_PRESETS = {
    'default': {
        'fields': {
            'delta': 0.05,
            'beta': 0.,
            'tau': 0.5,
            'r': 0.05,
            'k0': -0.1,
            'k1': 0.1,
            'k2': 4,
            'kl_sigma': kl_sigma,
            'kl_ratios': geom_kl_ratios,
            'N': 1e7,
            'w': 4e4,
            'D': 1.2e12,
            'labor_density': 8e6 * lognormal_distribution(6.5e5, kl_sigma, geom_kl_ratios),
        },
        'com': "System close to a Solow point, to which it rapidly converges",
        'plots': [],
    },
    
    'development': {
        'fields': {
            'delta': 0.05,
            'beta': 0.,
            'tau': 0.5,
            'r': 0.05,
            'k0': -0.1,
            'k1': 0.1,
            'k2': 4,
            'kl_sigma': kl_sigma,
            'kl_ratios': geom_kl_ratios,
            'N': 1e7,
            'w': 2.5e3,
            'D': 2e10,
            'labor_density': 2e6 * lognormal_distribution(2.5e3, kl_sigma, geom_kl_ratios),
        },
        'com': "System develops from a labor-intensive, low-wage start point",
        'plots': [],
    },
    
    'collapse': {
        'fields': {
            'delta': 0.05,
            'beta': 0.,
            'tau': 0.5,
            'r': 0.05,
            'k0': -0.1,
            'k1': 0.1,
            'k2': 4,
            'kl_sigma': kl_sigma,
            'kl_ratios': geom_kl_ratios,
            'N': 1e7,
            'w': 2.5e3,
            'D': 1e11,
            'labor_density': 2e6 * lognormal_distribution(2.5e3, kl_sigma, geom_kl_ratios),
        },
        'com': "System collapses due to the debt burden",
        'plots': [],
    }
}

_DPARAM = {
    # ---------
    # exogenous parameters
    # can also have a time variation (just replace by a function of time)
    # useful for studying the model's reaction to an exogenous shock
    'Tmax' : 500,
    'delta': 0.05,
    'beta': 0.,
    'tau': 0.5,
    'r': 0.05,
    'k0': -0.1,
    'k1': 0.1,
    'k2': 4,
    'kl_sigma': kl_sigma,
    'kl_ratios': geom_kl_ratios,

    # ---------
    # endogeneous functions

    # differential equations (ode)
    'N': {
        'func': lambda beta=0, itself=0: beta * itself,
        'initial': 1e7,
        'eqtype': 'ode',
    },
    'w': {
        'func': lambda L=0, N=1, itself=0, tau=1: (w0 * L / (N - L) - itself) / tau,
        'initial': 2.5e3,
        'eqtype': 'ode',
    },
    'D': {
        'func': lambda Pi=0, I=0: I - Pi,
        'initial': 1e11,
        'eqtype': 'ode',
    },
    # differential equations (pde)
    'labor_density': {
        'func': del_t_labor_density,
        'initial': 2e6 * lognormal_distribution(2.5e3, kl_sigma, geom_kl_ratios),
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
        'func': lambda GDP=0, w=0, L=0, r=0, D=0: GDP - w * L - r * D,
        'eqtype': 'intermediary',
    },
    'kappa': {
        'func': lambda k0=0, k1=0, k2=0, Pi=0, GDP=1: k0 + k1 * np.exp(k2*Pi/GDP) if Pi > 0 else 0,
        'eqtype': 'intermediary',
    },
    'I': {
        'func': lambda kappa=0, GDP=0: kappa * GDP,
        'eqtype': 'intermediary',
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
    },
    'd': {
        'func': lambda D=0, GDP=1: D / GDP,
        'eqtype': 'auxiliary',
    },
}
