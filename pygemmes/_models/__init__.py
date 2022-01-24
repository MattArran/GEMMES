# -*- coding: utf-8 -*-


import os
import importlib

from ._def_fields import _DFIELDS, _LIBRARY, _complete_DFIELDS
from ._def_functions import Funcs

_PATH_HERE = os.path.dirname(__file__)
_PATH_USER_HOME = os.path.expanduser('~')
_PATH_PRIVATE_MODELS = os.path.join(_PATH_USER_HOME, '.pygemmes', '_models')
_PATH_MODELS = _PATH_HERE


# if private pygemmes exists => load models from there
if not os.path.isdir(_PATH_PRIVATE_MODELS):
    _PATH_PRIVATE_MODELS = None


# ####################################################
# ####################################################
#       Automatically load all available models
# ####################################################


def _get_DMODEL(from_user=None):

    if from_user is None:
        from_user = True

    # ------------
    # path_model

    if from_user is True and _PATH_PRIVATE_MODELS is not None:
        path_models = _PATH_PRIVATE_MODELS
    else:
        path_models = _PATH_MODELS

    _df = {
        ff[:-3]: ff[len('_model_'):ff.index('.py')]
        for ff in os.listdir(path_models)
        if ff.startswith('_model_') and ff.endswith('.py')
    }

    _DMODEL = {}
    for k0, v0 in _df.items():
        pfe = os.path.join(path_models, k0 + '.py')
        spec = importlib.util.spec_from_file_location(k0, pfe)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        _DMODEL[v0] = {
            'logics': {k0: dict(v0) for k0, v0 in foo._LOGICS.items()},
            # 'func_order': foo._FUNC_ORDER,
            'file': foo.__file__,
            'description': foo.__doc__,
            'presets': {k0: dict(v0) for k0, v0 in foo._PRESETS.items()},
            'name': v0,
        }
    return path_models, _DMODEL


# ####################################################
# ####################################################
# Generic function to get the list of available models
# ####################################################


def get_available_models(
    model=None,
    details=None,
    returnas=None,
    verb=None,
    from_user=None,
):
    '''
    Check all models available in pygemmes, and gives back the information that are asked.
    With no arguments, it prints everything it can


    Parameters
    ----------
    model : Name of the models you are intereted in
    details : Boolean
    returnas : dict, none, list depending of what you need
    verb : print or not !
    from_user : TYPE, optional
    '''
    # -----------------
    # check inputs

    path_models, _DMODEL = _get_DMODEL(from_user=from_user)

    if model is None:
        model = sorted(_DMODEL.keys())
    if isinstance(model, str):
        model = [model]
    if returnas is None:
        returnas = False
    if verb is None:
        verb = returnas is False
    # -----------------
    # get available models

    dmod = {
        k0: {
            'file': str(_DMODEL[k0]['file']),
            'name': str(_DMODEL[k0]['name']),
            'description': str(_DMODEL[k0]['description']),
            'presets': list(_DMODEL[k0]['presets']),
        }
        for k0, v0 in _DMODEL.items()
        if k0 in model
    }

    if details is None:
        details = len(dmod) == 1

    # -----------------
    # print

    if verb is True or returnas is str:

        if details is True:
            # detailed message
            msg = "\n".join([
                "\n\n"
                f"{11*'#'}{'#'*len(v0['name'])}{11*'#'}\n"
                f"{10*'#'} {v0['name']} {10*'#'}\n"
                + v0['description']
                + "\n\n"
                + f"####Variables ####:\n"
                + f"# Ordinary differential equations ({len(_DMODEL[k0]['logics']['ode'])}):\n"
                + "\n".join([
                    f"\t-  {k1} :{(10-len(k1))*' '}{_DFIELDS.get(k1,{}).get('definition','')}"
                    for k1 in _DMODEL[k0]['logics']['ode']])
                + '\n'

                + f"# State Variables ({len(_DMODEL[k0]['logics']['statevar'])}):\n"
                + "\n".join([
                    f"\t-  {k1} :{(10-len(k1))*' '}{_DFIELDS.get(k1,{}).get('definition','')}"
                    for k1 in _DMODEL[k0]['logics']['statevar']])
                + '\n\n'
                + "\n####"
                + f"presets:\n"
                + "\n".join([
                    f" '{k1}' :"
                    f" {v1['com']}"
                    for k1, v1 in _DMODEL[k0]['presets'].items()
                ])


                + f"\n\nfile: {v0['file']}\n"
                for k0, v0 in dmod.items()
            ])

        else:
            # compact message
            nmax = max([len(v0['name']) for v0 in dmod.values()])
            lstr = [
                f"\t- {v0['name'].ljust(nmax)}: {v0['presets']}"
                for k0, v0 in dmod.items()
            ]
            msg = (
                f"The following models are available from '{path_models}'\n"
                + "\n".join(lstr)
            )

        if verb is True:
            print(msg)

    # -----------------
    # return

    if returnas is list:
        return model
    elif returnas is dict:
        return dmod
    elif returnas is str:
        return msg



def get_dfields_overview():


    # ----------------
    # get sub-dict of interest
    dparam_sub = _DFIELDS

    # ------------------
    # get column headers
    col2 = [
        'Field', 'definition', 'group', 'value', 'units',

    ]

    # ------------------
    # get values

    ar2 = [
        [k0,
         v0['definition'],
         v0['group'],
         v0['value'],
         v0['units'],

         ]
        for k0, v0 in dparam_sub.items()
    ]

    print(f'{len(dparam_sub)} fields in the library \n')

    return _utils._get_summary(
        lar=[ar2],
        lcol=[col2],
        verb=True,
        returnas=False,
    )
