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

l_kc_min, l_kc_max, n_kc = 2., 7., 1001
dl_kc = (l_kc_max - l_kc_min) / n_kc
geom_kc_ratios = 10**np.linspace(l_kc_min, l_kc_max, n_kc)
f_prefix, n = 300, 0.5
                   
# ---------------------------
# user-defined function order (optional)

_FUNC_ORDER = None

_DESCRIPTION = """
    DESCRIPTION: Model with wage growth determined by unemployment and with
        technologies with varying carbon intensity, fixed at creation time.
        A Phillips curve determines wage inflation phi_1 / (1 - u)**2 - phi_0,
        while the labor and carbon intensity of tech of capital-carbon ratio k
        are constant and 1 / f(k) = 1 / f0 (k/k0)^n, respectively, for
        f0 = $9k / tonne, k0 = $900 yr / tonne, n = 1/2.
        Emissions are taxed and regulated, profits are invested, and wages and
        taxes are consumed
    TYPICAL BEHAVIOUR: ?
    LINKTOARTICLE: Not yet written
    """

_PRESETS = {
    'default': {
        'fields': {
            'delta': 0.05,
            'beta': 0.,
            's_p': 1,
            's_w': 0,
            'kc_sigma': 0.1,
            'kc_ratios': geom_kc_ratios,
            'phinull': 1./6.,
            'N': 6e6,
            'w': 5e4,
            'labor_density': (5 / (10**dl_kc - 1)
                              * np.where(np.abs(np.log(geom_kc_ratios / 1e6)) <= 0.5 * dl_kc, 1, 0)),
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
            'kc_sigma': 0.1,
            'kc_ratios': geom_kc_ratios,
            'phinull': 1./6.,
            'N': 6e6,
            'w': 5e3,
            'labor_density': (1e6 / (np.log(2) * geom_kc_ratios * np.sqrt(2 * np.pi))
                              * np.exp(-0.5 * (np.log(geom_kc_ratios / 1e4) 
                                               / np.log(2))**2)),
        },
        'com': "System develops from a labor-intensive, low-wage start point",
        'plots': [],
    }
}

# ---------------------------
# user-defined model
# contains parameters and functions of various types


def del_t_carbon_intensity(I=0, I_distribution=np.ones(n_kc), delta=0,
                           kc_ratios=np.arange(1, n_kc + 1),
                           itself=np.zeros(n_kc)):
    return I * I_distribution / kc_ratios - delta * itself


def minimum_profitable_kc_ratio_index(kc_ratios=np.arange(n_kc), kc_min=0,
                                      productivity_fn=np.arange(n_kc),
                                      w=0, a=1, c_tax=0):
    if np.isscalar(w):
        return np.nonzero((kc_ratios >= kc_min)
                          & (productivity_fn * (1 - w / a) >= c_tax))[0][0]
    mpkri = [np.nonzero((kc_ratios >= kc_min)
                        & (productivity_fn * (1 - w_i / a_i) >= c_i))[0][0]
             if w_i <= a_i else kc_ratios.size
             for w_i, a_i, c_i in zip(w, a, c_tax)]
    return np.array(mpkri)

def profit_maximising_kc_ratio(kc_ratios=np.arange(1, n_kc + 1), kc_min=0,
                               productivity_fn=np.arange(n_kc), w=0, a=1,
                               c_tax = 0):
    if np.isscalar(w):
        return kc_ratios[np.argmax((kc_ratios >= kc_min)
                                   * (productivity_fn * (1 - w / a) - c_tax)
                                   / kc_ratios)]
    pmkr = [kc_ratios[np.argmax((kc_ratios >= kc_min)
                               * (productivity_fn * (1 - w_i / a_i) - c_i)
                               / kc_ratios)]
            for w_i, a_i, c_i in zip(w, a, c_tax)]
    return np.array(pmkr)


def lognormal_distribution(kc_optimum=1, kc_sigma=1,
                           kc_ratios=np.arange(1, n_kc + 1)):
    gaussian = (np.exp(-0.5 * np.log(kc_ratios / kc_optimum)**2 / kc_sigma**2)
                / kc_ratios)
    weighted_sum = (gaussian[:-1] * np.diff(kc_ratios)).sum()
    return np.pad(gaussian[:-1], (0, 1)) / weighted_sum


def total_production(id_kc_min=0, productivity_fn=np.arange(n_kc),
                     carbon_intensity=np.ones(n_kc),
                     kc_ratios=np.arange(1, n_kc + 1)):
    if np.isscalar(id_kc_min):
        id_kc_min = int(id_kc_min)
        return (productivity_fn[id_kc_min:-1] * carbon_intensity[id_kc_min:-1]
                * np.diff(kc_ratios[id_kc_min:])).sum()
    ltp = [(productivity_fn[i:-1] * ld[i:-1] * np.diff(kc_ratios[i:])).sum()
           for i, ld in zip(id_kc_min.astype(int), carbon_intensity)]
    return np.array(ltp)


def total_carbon(id_kc_min=0, carbon_intensity=np.ones(n_kc),
                 kc_ratios=np.arange(1, n_kc + 1)):
    if np.isscalar(id_kc_min):
        id_kc_min = int(id_kc_min)
        return (carbon_intensity[id_kc_min:-1] * np.diff(kc_ratios)).sum()
    ltl = [(ld[i:-1] * np.diff(kc_ratios[i:])).sum()
           for i, ld in zip(id_kc_min.astype(int), carbon_intensity)]
    return np.array(ltl)


_DPARAM = {
    # ---------
    # exogenous parameters
    # can also have a time variation (just replace by a function of time)
    # useful for studying the model's reaction to an exogenous shock
    'Tmax' : 50,
    'dt': 0.001,
    'beta': 0.,
    'delta': 0.05,
    'alpha': 0,
    'min_efficiency_ramp': 0,
    'tax_ramp': 0,
    's_p': 1,
    's_w': 0,
    'phinull': 1./6.,
    'kc_sigma': 0.1,
    'kc_ratios': geom_kc_ratios,

    # ---------
    # endogeneous functions

    # differential equations (ode)
    'N': {
        'func': lambda beta=0, itself=0: beta * itself,
        'initial': 6e6,
        'eqtype': 'ode',
    },
    'kc_min': {
        'func': lambda min_efficiency_ramp=0: min_efficiency_ramp,
        'initial': 0,
        'eqtype': 'ode',
    },
    'c_tax': {
        'func': lambda tax_ramp=0: tax_ramp,
        'initial': 2e3,
        'eqtype': 'ode',
    },
    'w': {
        'func': lambda itself=0, phillips=0: itself * phillips,
        'initial': 1.2e5,
        'eqtype': 'ode',
    },
    'a': {
        'func': lambda itself=0, alpha=0: itself * alpha,
        'initial': 2e5,
        'eqtype': 'ode',
    },
    # differential equations (pde)
    'carbon_intensity': {
        'func': del_t_carbon_intensity,
        'initial': (1e8 / (np.log(2) * geom_kc_ratios * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * (np.log(geom_kc_ratios / 4e2) 
                             / np.log(2))**2)),
        'eqtype': 'pde',
    },

    # Intermediary functions (endogenous, computation intermediates)
    'productivity_fn': {
        'func': lambda kc_ratios=np.arange(n_kc): f_prefix * kc_ratios**n,
        'eqtype': 'intermediary',
    },
    'id_kc_min': {
        'func': minimum_profitable_kc_ratio_index,
        'eqtype': 'intermediary',
    },
    'kc_optimum': {
        'func': profit_maximising_kc_ratio,
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
    'C': {
        'func': total_carbon,
        'eqtype': 'intermediary',
    },
    'L': {
        'func': lambda GDP=0, a=1: GDP / a,
        'eqtype': 'intermediary',
    },
    'Pi': {
        'func': lambda GDP=0, w=0, L=0, c_tax=0, C=0: GDP - w * L - c_tax * C,
        'eqtype': 'intermediary',
    },
    'I': {
        'func': lambda Pi=0: Pi,
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
        'func': lambda Pi=0, GDP=1: 1 - Pi / GDP,
        'eqtype': 'auxiliary',
    }
}
