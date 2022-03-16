# -*- coding: utf-8 -*-

"""
This file contains the default fields (units, dimension, symbol...) for all
common parameters / variables that can be used by any model.

It can be used a common database where all default fields attached to each
parameter / variable are stored

Users can decide to replace some fields when they define their model, but all
fields which are not explicitly described by the user / modeller in the model
will be taken from this default database

----

This file contains :
    _DFIELDS the big dictionnary with basic information on
        * Variables
        * Parameters
        * Numerical parameters

    _DALLOWED_FIELDS : Contains all the restrictions on each field for each
    element in _DFIELD
    _DEFAULTFIELDS : The value that will be added if none are

    __DOTHECHECK Flag to check or not the dictionnary
    __FILLDEFAULTVALUES Flag to fill the defaultfields
"""


import warnings


import numpy as np

__DOTHECHECK = False
__FILLDEFAULTVALUES = True

# #############################################################################
# #############################################################################
#                   FIELDS OF FIELDS AND EXPECTED VALUES
# #############################################################################
# dict of default value in fields
_DEFAULTFIELDS = {
    'value': None,
    'com': 'No comment',
    'dimension': 'undefined',
    'units': 'undefined',
    'type': 'undefined',
    'group': None,
    # 'symbol' : this is the key of the variable
}

# dict of allowed fields (None => no restriction)
_DALLOWED_FIELDS = {
    'value': None,
    'com': None,
    'units': [
        'Real Units',  #
        'yr',      # Time
        'Dollars',      # Money
        'C',      # Concentration
        'Humans',  # Population
    ],
    'type': ['intensive', 'extensive', 'dimensionless'],
    'symbol': None,
    'group': None,  # [
    # 'Numerical',
    # 'Population',
    # 'Prices', 'Capital', 'Philips', 'Gemmes',
    # 'Keen', 'Dividends', 'Economy', 'Production',
    # 'Coupling',
    # 'RelaxBuffer',
    # 'Climate', 'Damage',
    # ],
}

# #############################################################################
# #############################################################################
#                   Dict of default fields
# #############################################################################


_DFIELDS = {

    # --------------
    # Numerical
    'Tmax': {
        'value': 100,
        'com': 'Duration of simulation',
        'dimension': 'time',
        'units': 'yr',
        'type': None,
        'symbol': None,
        'group': 'Numerical',
    },
    'dt': {
        'value': 0.01,
        'com': 'Time step (fixed timestep method)',
        'dimension': 'time',
        'units': 'yr',
        'type': None,
        'symbol': None,
        'group': 'Numerical',
    },
    'nt': {
        'func': lambda Tmax=0, dt=1: int(Tmax / dt),  # Dynamically allocated
        'eqtype': 'intermediary',
        'com': 'Number of temporal iteration',
        'dimension': None,
        'units': None,
        'type': None,
        'symbol': None,
        'group': 'Numerical',
    },
    'nx': {
        'value': 1,
        'com': 'Number of similar systems evolving in parrallel',
        'dimension': None,
        'units': None,
        'type': None,
        'symbol': None,
        'group': 'Numerical',
    },


    # --------------
    # Time vector
    'time': {
        'func': lambda dt=0: 1.,
        'com': 'Time vector',
        'dimension': 'time',
        'units': 'yr',
        'type': 'extensive',
        'symbol': r'$t$',
        'group': 'Time',
        'eqtype': 'ode',
        'initial': 0,
    },
    'tau': {
        'value': 1,
        'com': 'Adjustment timescale',
        'dimension': 'time',
        'units': 'yr',
        'type': 'intensive',
        'symbol': r'$\tau$',
        'group': 'Time',
    },

    # PARAMETERS #############################################################
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # Population evolution
    'beta': {
        'value': 0.025,
        'com': 'Rate of population growth',
        'dimension': 'time rate',
        'units': 'yr^{-1}',
        'type': 'intensive',
        'symbol': r'$beta$',
        'group': 'Population',
    },
    'alpha': {
        'value': 0.02,
        'com': 'Rate of productivity increase',
        'dimension': 'time rate',
        'units': 'yr^{-1}',
        'type': 'intensive',
        'symbol': r'$alpha$',
        'group': 'Population',
    },

    # --------------
    # Capital properties
    'delta': {
        'value': 0.005,
        'com': 'Rate of capital depletion',
        'dimension': 'time rate',
        'units': 'yr^{-1}',
        'type': 'intensive',
        'symbol': r'$\delta$',
        'group': 'Capital',
    },
    's_p': {
        'value': 1,
        'com': 'Proportion of profit invested in capital',
        'dimension': '',
        'units': '',
        'type': 'intensive',
        'symbol': r'$s_p$',
        'group': 'Capital',
    },
    's_w': {
        'value': 0,
        'com': 'Proportion of wages invested in capital',
        'dimension': '',
        'units': '',
        'type': 'intensive',
        'symbol': r'$s_w$',
        'group': 'Capital',
    },

    # --------------
    # Production
    'nu': {
        'value': 3,
        'com': 'Kapital to output ratio',
        'dimension': None,
        'units': None,
        'type': 'intensive',
        'symbol': r'\nu',
        'group': 'Production',
    },

    # --------------
    # INTEREST / Price
    'r': {
        'value': .03,
        'com': 'Interest at the bank',
        'dimension': 'time rate',
        'units': 'yr^{-1}',
        'type': 'intensive',
        'symbol': None,
        'group': 'Prices',
    },

    # --------------
    # PHILIPS CURVE (employement-salary increase)
    'phinull': {
        'value': 0.04,
        'com': 'Unemployment rate that stops salary increase (no inflation)',
        'dimension': None,
        'units': None,
        'type': 'intensive',
        'symbol': r'$\phi_0$',
        'group': 'Philips',
    },
    'phi0': {
        'value': None,
        'com': '',
        'dimension': None,
        'units': None,
        'type': '',
        'symbol': r'$\phi_0$',
        'group': 'Philips',
    },
    'phi1': {
        'value': None,
        'com': '',
        'dimension': None,
        'units': None,
        'type': '',
        'symbol': r'$\phi_0$',
        'group': 'Philips',
    },

    # --------------
    # KEEN INVESTMENT FUNCTION (profit-investment function)
    'k0': {
        'value': -0.0065,
        'com': 'Percent of GDP invested when profit is zero',
        'dimension': None,
        'units': None,
        'type': 'intensive',
        'symbol': r'$k_0$',
        'group': 'Keen',
    },
    'k1': {
        'value': np.exp(-5),
        'com': 'Investment slope',
        'dimension': None,
        'units': None,
        'type': 'intensive',
        'symbol': r'$k_1$',
        'group': 'Keen',
    },
    'k2': {
        'value': 20,
        'com': 'Investment power in kappa',
        'dimension': None,
        'units': None,
        'type': 'intensive',
        'symbol': r'$k_2$',
        'group': 'Keen',
    },

    # --------------
    # PUTTY-CLAY MODEL FOR CAPITAL INVESTMENT
    'kl_ratios': {
        'value': np.arange(100),
        'com': 'Possible capital-labor ratios',
        'dimension': 'Capital-labor ratio',
        'units': 'Dollars Humans^{-1}',
        'type': 'intensive',
        'symbol': r'$\sigma_{kl}$',
        'group': 'Putty-Clay',
    },
    'productivity_fn': {
        'value': np.arange(100),
        'com': 'Labor productivity as a function of capital-labor ratio',
        'dimension': 'Labor productivity',
        'units': 'Dollars Human^{-1} yr^{-1}',
        'type': 'intensive',
        'symbol': r'$f$',
        'group': 'Production',
    },
    'labor_density': {
        'value': np.zeros(100),
        'com': 'Labor employable on machines of varying capital-labor ratio',
        'dimension': 'Labor density',
        'units': 'Humans^{2] Dollars^{-1}',
        'type': 'extensive',
        'symbol': r'$l$',
        'group': 'Putty-Clay',
    },
    'id_kl_min': {
        'value': 0,
        'com': 'ID of minimum capital-labor ratio for operation',
        'dimension': '',
        'units': 'Array index',
        'type': 'intensive',
        'symbol': r'$k_{min}$',
        'group': 'Putty-Clay',
    },
    'kl_optimum': {
        'value': 0,
        'com': 'Profit-optimal capital-labor ratio',
        'dimension': 'Capital-labor ratio',
        'units': 'Dollars Humans^{-1}',
        'type': 'intensive',
        'symbol': r'$k_{opt}$',
        'group': 'Putty-Clay',
    },
    'kl_bar': {
        'value': 0,
        'com': 'Mean available capital-labor ratio',
        'dimension': 'Capital-labor ratio',
        'units': 'Dollars Humans^{-1}',
        'type': 'intensive',
        'symbol': r'$\bar{k}$',
        'group': 'Putty-Clay',
    },
    'kl_sigma': {
        'value': 0.1,
        'com': 'Spread of investment around optimal capital-labor ratio',
        'dimension': '',
        'units': '',
        'type': 'intensive',
        'symbol': r'$\sigma_{kl}$',
        'group': 'Putty-Clay',
    },
    'I_distribution': {
        'value': np.ones(100),
        'com': 'Distribution of investment around optimal capital-labor ratio',
        'dimension': 'Labor-capital ratio',
        'units': 'Humans Dollar^{-1}',
        'type': 'intensive',
        'symbol': r'$\delta_{D}$',
        'group': 'Putty-Clay',
    },
    
    # --------------
    # PUTTY-CLAY MODEL FOR CARBON INTENSITY
    'kc_ratios': {
        'value': np.arange(100),
        'com': 'Possible capital-carbon emission ratios',
        'dimension': 'Capital-carbon ratio',
        'units': 'Dollars yr Tonnes^{-1}',
        'type': 'intensive',
        'symbol': r'$\sigma_{kc}$',
        'group': 'Putty-Clay_carbon',
    },
    'carbon_intensity': {
        'value': np.zeros(100),
        'com': 'Potential carbon emissions by machines of varying capital-carbon ratio',
        'dimension': 'Carbon intensity',
        'units': 'Tonnes yr^{-1}',
        'type': 'extensive',
        'symbol': r'$l$',
        'group': 'Putty-Clay_carbon',
    },
    'id_kc_min': {
        'value': 0,
        'com': 'ID of minimum capital-carbon ratio for operation',
        'dimension': '',
        'units': 'Array index',
        'type': 'intensive',
        'symbol': r'index$(k_{min})$',
        'group': 'Putty-Clay_carbon',
    },
    'kc_min': {
        'value': 0,
        'com': 'Regulated minimum capital-carbon ratio',
        'dimension': 'Capital-carbon ratio',
        'units': 'Dollars yr Tonnes^{-1}',
        'type': 'intensive',
        'symbol': r'$k_{min}$',
        'group': 'Putty-Clay_carbon',
    },
    'kc_optimum': {
        'value': 0,
        'com': 'Profit-optimal capital-carbon ratio',
        'dimension': 'Capital-carbon ratio',
        'units': 'Dollars yr Tonnes^{-1}',
        'type': 'intensive',
        'symbol': r'$k_{opt}$',
        'group': 'Putty-Clay_carbon',
    },
    'kc_bar': {
        'value': 0,
        'com': 'Mean available capital-carbon ratio',
        'dimension': 'Capital-carbon ratio',
        'units': 'Dollars yr Tonnes^{-1}',
        'type': 'intensive',
        'symbol': r'$\bar{k}$',
        'group': 'Putty-Clay_carbon',
    },
    'kc_sigma': {
        'value': 0.1,
        'com': 'Spread of investment around optimal capital-carbon ratio',
        'dimension': '',
        'units': '',
        'type': 'intensive',
        'symbol': r'$\sigma_{kl}$',
        'group': 'Putty-Clay_carbon',
    },
    'min_efficiency_ramp': {
        'value': 0,
        'com': 'Rate of regulatory minimum increase',
        'dimension': 'Dollars Tonnes^{-1}',
        'units': 'Dollars Tonnes^{-1}',
        'type': 'intensive',
        'symbol': r'$d_t k_{min}$',
        'group': 'Putty-Clay_carbon',
    },
    'tax_ramp': {
        'value': 0,
        'com': 'Rate of carbon tax increase',
        'dimension': 'Dollars yrs^{-1}',
        'units': 'Dollars yrs^{-1}',
        'type': 'intensive',
        'symbol': r'$c_t$',
        'group': 'Putty-Clay_carbon',
    },
    'c_tax': {
        'value': 0,
        'com': 'Tax imposed on carbon emissions',
        'dimension': 'Dollars Tonnes^{-1}',
        'units': 'Dollars Tonnes^{-1}',
        'type': 'intensive',
        'symbol': r'$c$',
        'group': 'Putty-Clay_carbon',
    },
    'C': {
        'value': 0,
        'com': 'Total carbon emissions',
        'dimension': 'Tonnes',
        'units': 'Tonnes',
        'type': 'intensive',
        'symbol': r'$C$',
        'group': 'Putty-Clay_carbon',
    },
    # --------------
    # TWO-TRACK LABOR MARKET
    'labor_effectiveness': {
        'value': np.ones([2, 100]),
        'com': 'Labor class effectiveness as a function of capital-labor ratio',
        'dimension': '',
        'units': '',
        'type': 'intensive',
        'symbol': r'$\epsilon$',
        'group': 'Two-track labor',
    },
    'labor_employed': {
        'value': np.ones([2, 100]),
        'com': 'Booleans for labor class employment on varying technology',
        'dimension': '',
        'units': '',
        'type': 'intensive',
        'symbol': r'$\iota$',
        'group': 'Two-track labor',
    },
    'tau_delta': {
        'value': 50,
        'com': 'Timescale for technical progress',
        'dimension': 'time',
        'units': 'yr',
        'type': 'intensive',
        'symbol': r'$\tau_\delta$',
        'group': 'Two-track labor',
    },

    # DYNAMICAL VARIABLES ####################################################
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # Classic dimensionless for phase-space
    'omega': {
        'value': .578,
        'com': 'Wage share of the economy',
        'dimension': '',
        'units': '',
        'type': 'dimensionless',
        'symbol': r'$\omega$',
        'group': 'Economy',
    },
    'lambda': {
        'value': .675,
        'com': 'employment rate',
        'dimension': '',
        'units': '',
        'type': 'dimensionless',
        'symbol': r'$\lambda$',
        'group': 'Economy',
    },

    'd': {
        'value': 1.53,
        'com': 'relative private debt',
        'dimension': '',
        'units': '',
        'type': 'dimensionless',
        'symbol': r'$d$',
        'group': 'Economy',
    },

    # Intensive dynamic variable
    'a': {
        'value': None,
        'com': 'productivity per worker',
        'dimension': 'Productivity',
        'units': 'Real Units Humans^{-1} yr^{-1}',
        'type': 'intensive',
        'symbol': r'$a$',
        'group': 'Economy',
    },
    'N': {
        'value': 4.83,
        'com': 'Population',
        'dimension': 'Humans',
        'units': 'Humans',
        'type': 'extensive',
        'symbol': r'$N$',
        'group': 'Population',
    },


    'K': {
        'value': None,
        'com': 'Capital',
        'dimension': 'Real Units',
        'units': 'Real Units',
        'type': 'extensive',
        'symbol': r'$K$',
        'group': 'Economy',
    },
    'w': {
        'value': None,
        'com': 'Salary',
        'dimension': 'Money',
        'units': 'Dollars Humans^{-1} yr^{-1}',
        'type': 'extensive',
        'symbol': r'$w$',
        'group': 'Economy',
    },
    'D': {
        'value': None,
        'com': 'Absolute private debt',
        'dimension': 'Money',
        'units': 'Dollars',
        'type': 'extensive',
        'symbol': r'$D$',
        'group': 'Economy',
    },

    # INTERMEDIARY VARIABLES #################################################
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    # From functions
    'kappa': {
        'value': None,
        'com': 'Part of GDP in investment',
        'dimension': '',
        'units': '',
        'type': 'dimensionless',
        'symbol': r'$\kappa$',
        'group': 'Economy',
    },
    'phillips': {
        'value': None,
        'com': 'Wage inflation rate',
        'dimension': 'time rate',
        'units': 'yr^{-1}',
        'type': 'intensive',
        'symbol': r'$\phi$',
        'group': 'Economy',
    },
    'pi': {
        'value': None,
        'com': 'relative profit',
        'dimension': '',
        'units': '',
        'type': 'dimensionless',
        'symbol': r'$\pi$',
        'group': 'Economy',
    },
    'g': {
        'value': None,
        'com': 'Relative growth',
        'dimension': 'time rate',
        'units': 'yr^{-1}',
        'type': 'intensive',
        'symbol': r'$g$',
        'group': 'Economy',
    },
    'GDP': {
        'value': None,
        'com': 'GDP in nominal term',
        'dimension': 'Money',
        'units': 'Dollars yr^{-1}',
        'type': 'extensive',
        'symbol': r'$GDP$',
        'group': 'Economy',
    },
    'Y': {
        'value': None,
        'com': 'GDP in output quantity',
        'dimension': 'Real Units',
        'units': 'Real units yr^{-1}',
        'type': 'extensive',
        'symbol': r'$Y$',
        'group': 'Missing',
    },
    'L': {
        'value': None,
        'com': 'Workers',
        'dimension': 'Humans',
        'units': 'Humans',
        'type': 'extensive',
        'symbol': r'$L$',
        'group': 'Missing',
    },
    'I': {
        'value': None,
        'com': 'Investment',
        'dimension': 'Money',
        'units': 'Dollars yr^{-1}',
        'type': 'extensive',
        'symbol': r'$I$',
        'group': 'Economy',
    },
    'Pi': {
        'value': None,
        'com': 'Absolute profit',
        'dimension': 'Money',
        'units': 'Dollars yr^{-1}',
        'type': 'extensive',
        'symbol': r'$\Pi$',
        'group': 'Economy',
    },

    'i': {
        'value': None,
        'com': 'Inflation rate',
        'dimension': 'time rate',
        'units': 'yr^{-1}',
        'type': 'intensive',
        'symbol': r'$i$',
        'group': 'Economy',
    },
}


# #############################################################################
# #############################################################################
#               Conformity checks (for saefty, to detect typos...)
# #############################################################################


def Complete_DFIELDS(_DFIELDS, _DEFAULTFIELDS):
    for k in _DFIELDS.keys():
        kkey = _DFIELDS[k].keys()
        basekeys = ['value',
                    'dimension',
                    'symbol',
                    'com',
                    'units',
                    'type',
                    'group']
        for v in basekeys:
            if v not in kkey:
                _DFIELDS[k][v] = _DEFAULTFIELDS[v]
        if 'symbol' not in kkey:
            _DFIELDS[k]['symbol'] = k
    return _DFIELDS


def Check_DFIELDS(_DFIELDS, _DALLOWED_FIELDS):
    # List non-conform keys in dict, for detailed error printing
    dk0 = {
        k0: [
            v0[ss] for ss in _DALLOWED_FIELDS.keys()
            if _DALLOWED_FIELDS[ss] is not None and not (
                v0[ss] is None
                or v0[ss] == ''
                or v0[ss] in _DALLOWED_FIELDS[ss]
            )
        ]
        for k0, v0 in _DFIELDS.items()
        if not (
            isinstance(v0, dict)
            and sorted(_DALLOWED_FIELDS) == sorted(v0.keys())
            and all([
                v0[ss] is None
                or v0[ss] == ''
                or v0[ss] in _DALLOWED_FIELDS[ss]
                for ss in _DALLOWED_FIELDS.keys()
                if _DALLOWED_FIELDS[ss] is not None
            ])
        )
    }

    # Raise warning if any non-conformity
    # Include details per key
    if len(dk0) > 0:
        lstr = [f'\t- {k0}: {v0}' for k0, v0 in dk0.items()]
        msg = (
            "The following keys of _DFIELDS are non-conform:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)


if __FILLDEFAULTVALUES:
    _DFIELDS = Complete_DFIELDS(_DFIELDS, _DEFAULTFIELDS)
if __DOTHECHECK:
    Check_DFIELDS(_DFIELDS, _DALLOWED_FIELDS)
