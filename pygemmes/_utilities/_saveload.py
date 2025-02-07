# -*- coding: utf-8 -*-


# Built-in
import os
import inspect
import itertools as itt
import getpass
import importlib
import datetime as dtm
import warnings


# common
import numpy as np


_PATH_HERE = os.path.dirname(__file__)
_PATH_USER_HOME = os.path.expanduser('~')
_PATH_PRIVATE_OUTPUT = os.path.join(_PATH_USER_HOME, '.pygemmes', 'output')
if os.path.isdir(_PATH_PRIVATE_OUTPUT):
    _PATH_OUTPUT = _PATH_PRIVATE_OUTPUT
else:
    _PATH_OUTPUT = os.path.join(os.path.dirname(_PATH_HERE), 'output')


_PATH_MODELS = os.path.join(os.path.dirname(_PATH_HERE), '_models')
_PATH_PRIVATE_MODELS = os.path.join(_PATH_USER_HOME, '.pygemmes', '_models')


_INCLUDE_NAME = ['model', 'preset', 'solver', 'name', 'user', 'date']


# #############################################################################
# #############################################################################
#                               save
# #############################################################################


def _check_files(pfe):
    """ check files validity, and propose corrected files path if possible """

    try:
        pfe = _check_files_validity(pfe)
    except Exception as err:
        # if the files are not valid
        # it may be because the user just provided the file name without the
        # path => add the path and try again
        if isinstance(pfe, str):
            pfe = [pfe]
        pfe = [os.path.join(_PATH_OUTPUT, ff) for ff in pfe]
        pfe = _check_files_validity(pfe)
    return pfe


def _check_files_validity(pfe):
    """ Check the provided file names are valid """

    if isinstance(pfe, str):
        pfe = [pfe]
    if not isinstance(pfe, list):
        msg = f"Arg pfe must be a list of str!\nProvided:\n\t{pfe}"
        raise Exception(msg)

    lfout = [
        ff for ff in pfe
        if not (
            isinstance(ff, str)
            and os.path.isfile(ff)
            and os.path.split(ff)[-1].startswith('Output_')
            and ff.endswith('.npz')
            and len(os.path.split(ff)[-1].split('_')) == len(_INCLUDE_NAME) + 1
        )
    ]
    if len(lfout) > 0:
        lstr = [f'\t- {ff}' for ff in lfout]
        msg = (
            "Non-conform files detected:\n"
            + "\n".join(lstr)
            + "\nThe file names should:\n"
            + "\t- be of the form 'Output_....npz'\n"
            + "\t- include the following components, separated by '_':"
            + f"{_INCLUDE_NAME}"
        )
        raise Exception(msg)
    return pfe


# #############################################################################
# #############################################################################
#                               save
# #############################################################################


def _save_check_inputs(
    path=None,
    name=None,
    fmt=None,
    verb=None,
    returnas=None,
):

    # path
    if path is None:
        path = _PATH_OUTPUT
    if not os.path.isdir(path):
        msg = (
            "Arg path must be a valid path!\n"
            f"Provided: {path}"
        )
        raise Exception(msg)
    path = os.path.abspath(path)

    # name
    if name is None:
        name = 'placeholder'
    if not isinstance(name, str):
        msg = (
            "Arg name must be a str!\n"
            f"Provided: {name}"
        )
        raise Exception(msg)
    name = name.replace('_', '-').replace(' ', '-')

    # fmt
    if fmt is None:
        fmt = 'npz'
    lfmt = ['npz']
    if fmt not in lfmt:
        msg = (
            "Arg fmt must be a valid format!\n"
            f"\t- provided: {fmt}\n"
            f"\t- allowed: {lfmt}"
        )
        raise Exception(msg)

    # verb
    if verb is None:
        verb = True
    if not isinstance(verb, bool):
        msg = (
            "Arg verb must be a bool!\n"
            f"\t- provided: {verb}"
        )
        raise Exception(msg)

    # returnas
    if returnas is None:
        returnas = False
    if returnas not in [str, False]:
        msg = (
            "Arg returnas must be a in [str, False]!\n"
            f"\t- provided: {returnas}"
        )
        raise Exception(msg)

    return path, name, fmt, verb, returnas


def save(
    obj,
    path=None,
    name=None,
    fmt=None,
    verb=None,
    returnas=None,
):

    # ------------
    # check inputs
    path, name, fmt, verb, returnas = _save_check_inputs(
        path=path,
        name=name,
        fmt=fmt,
        verb=verb,
        returnas=returnas,
    )

    # -------------
    # set extension
    if fmt == 'npz':
        ext = 'npz'

    # --------------
    # Make full name
    dinclude = {
        'model': obj.dmodel['name'].replace('_', '-').replace(' ', '-'),
        'preset': obj.dmodel['preset'],
        'solver': obj.dmisc['solver'],
        'name': name,
        'user': getpass.getuser(),
        'date': dtm.datetime.utcnow().strftime('%Y%m%d-%H%M%S'),
    }

    lk = list(dinclude.keys())
    for k0 in lk:
        if k0 not in _INCLUDE_NAME:
            del dinclude[k0]

    fullname = 'Output'
    for k0, v0 in dinclude.items():
        fullname += '_{}'.format(v0)
    fullname += f'.{ext}'

    # --------------
    # convert data for saving

    if fmt == 'npz':
        # convert obj to dict
        dout = obj._to_dict()

        # delete lambda functions
        # they can't be saved, and will be reloaded from the model file
        for k0 in dout['dparam'].keys():
            if dout['dparam'][k0].get('func') is not None:
                dout['dparam'][k0]['func'] = True

        # the model source file stored in self.model

        # full saving pfe (Path-File-Extension)
        pfe = os.path.join(path, fullname)
        np.savez(pfe, **dout)

    # --------------
    # verb and return
    if verb is True:
        msg = (f"Saved in:\n\t{pfe}")
        print(msg)

    if returnas is str:
        return pfe


# #############################################################################
# #############################################################################
#                              load
# #############################################################################


def load(pfe, model_file=None):
    """ Load a save output file

    model_file can be used to specify the model file from which to load the
    non-lambda functions that need to be reconstructed

    Only necessary if the original model file cannot be found automatically

    """

    # --------------
    # check inputs
    pfe = _check_files(pfe)

    # --------------
    # load and instanciate
    from .. import _core

    lobj = [
        _core.Hub._from_dict(
            {
                k0: v0.tolist()
                for k0, v0 in dict(np.load(ff, allow_pickle=True)).items()
            },
            model_file=model_file,
        )
        for ff in pfe
    ]
    return lobj


# #############################################################################
# #############################################################################
#           Rebuild function from source (for loading)
# #############################################################################


def rebuild_func_from_source(dout=None, model_file=None):
    """ Rebuild functions (that can't be saved) for saved source

    When saving a Hub, the functions are lost
    They have to be re-built from source at loading time
    """

    for k0, v0 in dout['dparam'].items():
        if dout['dparam'][k0].get('func') is not None:

            if dout['dparam'][k0].get('source_exp') is None:

                # Rebuild from source file
                mod = dout['dmodel']['name']
                fname = dout['dparam'][k0]['source_name']
                if isinstance(model_file, str):
                    if not os.path.isfile(model_file):
                        msg = (
                            "Provided model_file does not exist:\n"
                            f"{model_file}"
                        )
                        raise Exception(msg)
                    pfe = model_file
                else:
                    pfe = dout['dmodel']['file']
                    if not os.path.isfile(pfe):
                        # try loading from local output ref (for unit tests)
                        path, tail = os.path.split(pfe)
                        if '.pygemmes' in path:
                            pfe1 = os.path.join(_PATH_PRIVATE_MODELS, tail)
                        else:
                            pfe1 = os.path.join(_PATH_MODELS, tail)
                        if not os.path.isfile(pfe1):
                            lstr = [f'\t- {pp}' for pp in [pfe, pfe1]]
                            msg = (
                                f"Function {k0} could not be re-loaded from:\n"
                                + f"\n".join(lstr)
                            )
                            raise Exception(msg)
                        pfe = pfe1

                spec = importlib.util.spec_from_file_location(mod, pfe)
                foo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(foo)

                c0 = (
                    hasattr(foo, fname)
                    and hasattr(getattr(foo, fname), '__call__')
                )
                if not c0:
                    msg = (
                        f"Module {mod} has no callable attribute "
                        f"named {fname}\n"
                        f"file: {pfe}"
                    )
                    raise Exception(msg)
                dout['dparam'][k0]['func'] = getattr(foo, fname)

            else:
                # Rebuild from source str
                kargs = dout['dparam'][k0]['source_kargs']
                dout['dparam'][k0]['func'] = eval(
                    f"lambda {kargs}: {dout['dparam'][k0]['source_exp']}"
                )[0]


# #############################################################################
# #############################################################################
#                   Parse file names
# #############################################################################


def get_file_parsing(pfe=None):

    # ---------
    # check inputs
    pfe = _check_files(pfe)

    # ------------
    # parse

    df = dict.fromkeys(pfe)
    for ff in pfe:
        fn = os.path.split(ff)[-1]
        fmt = fn.split('.')[-1]
        fn = fn[:fn.index(f'.{fmt}')]
        model, preset, solver, name, user, date = fn.split('_')[1:]
        df[ff] = {
            'model': model,
            'preset': preset,
            'solver': solver,
            'name': name,
            'user': user,
            'date': date,
            'fmt': fmt,
        }
    return df


def _get_available_output_check(
    path=None,
    model=None,
    preset=None,
    user=None,
    solver=None,
    name=None,
    fmt=None,
    verb=None,
    returnas=None,
):
    # ----
    # path

    if path is None:
        path = _PATH_OUTPUT
    if not (isinstance(path, str) and os.path.isdir(path)):
        msg = (
            "Arg path must be a valid directory!\n"
            f"Provided:\n\t{path}"
        )
        raise Exception(msg)

    # --------
    # criteria

    dcrit = {
        'model': model,
        'preset': preset,
        'user': user,
        'solver': solver,
        'name': name,
        'fmt': fmt,
    }

    dfail = {}
    lk = list(dcrit.keys())
    for k0 in lk:

        # remove if non-relevant
        if dcrit[k0] is None:
            del dcrit[k0]
            continue

        # check generic validity
        if not isinstance(dcrit[k0], str):
            msg = "must be a str!"
            dfail[k0] = msg
            continue

        # check particular cases
        if k0 == 'fmt' and dcrit[k0] not in ['npz']:
            msg = f"not valid value ({dcrit[k0]} vs ['npz']"
            dfail[k0] = msg
            continue

        # format model and name
        if k0 in ['model', 'preset', 'name']:
            dcrit[k0] = dcrit[k0].replace('_', '-').replace(' ', '-')

    if len(dfail) > 0:
        lstr = [f'\t- {k0}: {v0}' for k0, v0 in dfail.items()]
        msg = (
            "The foolowing criteria are non-valid:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ----------
    # returnas
    if returnas is None:
        returnas = False
    if returnas not in [False, list, dict]:
        msg = (
            f"Arg returnas must be either:\n"
            "\t- False: no return\n"
            "\t- list: return list of file names\n"
            "\t- dict: for each file also return its parsed components\n"
        )
        raise Exception(msg)

    # ----------
    # verb
    if verb is None:
        verb = returnas is False
    if not isinstance(verb, bool):
        msg = f"Arg verb must be a bool!\nProvided: {verb}"
        raise Exception(msg)

    return path, dcrit, verb, returnas


def get_available_output(
    path=None,
    model=None,
    preset=None,
    user=None,
    solver=None,
    name=None,
    fmt=None,
    verb=None,
    returnas=None,
):

    # ------------
    # Check inputs

    path, dcrit, verb, returnas = _get_available_output_check(
        path=path,
        model=model,
        preset=preset,
        user=user,
        solver=solver,
        name=name,
        fmt=fmt,
        verb=verb,
        returnas=returnas,
    )

    # -------------
    # get list of files

    lf = [
        os.path.join(path, ff)
        for ff in os.listdir(path)
        if ff.startswith('Output_') and ff.endswith('.npz')
    ]

    # -----------
    # get parsing
    df = get_file_parsing(pfe=lf)

    # -----------
    # filter if relevant
    for ff in lf:
        for k0, v0 in dcrit.items():
            if ff in df.keys():
                if df[ff][k0] != v0:
                    del df[ff]
    lf = list(df.keys())

    # -----------
    # verb

    if verb is True:
        lcrit = [f'\t- {k0}: {v0}' for k0, v0 in dcrit.items()]
        if len(lcrit) > 0:
            lcrit[-1] += "\n"
        lstr = [f'\t- {ff}' for ff in lf]
        msg = (
            "File selection criteria:\n"
            + f"\t- path: {path}\n"
            + "\n".join(lcrit)
            + "\nAvailable files matching criteria:\n"
            + "\n".join(lstr)
        )
        print(msg)

    # -----------
    # return

    if returnas is list:
        return lf
    elif returnas is dict:
        return df
