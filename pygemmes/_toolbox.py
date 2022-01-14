# -*- coding: utf-8 -*-

# Here we decide what the user will see
from ._core import Hub
from ._models import get_available_models, get_dfields_overview
from ._utilities._solvers import get_available_solvers
from ._utilities._saveload import get_available_output, load
from ._models import get_available_models, _DFIELDS
from pygemmes._utilities import _utils


import matplotlib.pyplot as plt
import numpy as np

# #############################################################################

__all__ = [
    'get_dfields_overview',
    'create_preset_from_model_preset',
    'plot_one_run_all_solvers',
    'comparesolver_Lorenz',
    'testConvergence_DampOsc',
    'Generate_network_logics',
    'GenerateIndividualSensitivity',
    'GenerateCoupledSensitivity'
]


def get_dfields_overview(returnas=False):
    '''
    Will load the library of fields, then all available models,
    and will print the ensemble of fields with their properties and the models they are in.

    returnas can be used on "list,dict"

    if a field has no group, then it is only defined in model files
    '''
    dparam_sub = _DFIELDS
    for key, val in dparam_sub.items():
        dparam_sub[key]['inmodel'] = []
    models = get_available_models(returnas=list)

    fieldsnotinDfields = []
    for model in models:
        # print(model)
        hub = Hub(model, verb=False)
        params = hub.get_dparam(returnas=dict)
        for key in params.keys():
            if key in dparam_sub:
                if 'inmodel' not in dparam_sub[key].keys():
                    dparam_sub[key]['inmodel'] = []
                dparam_sub[key]['inmodel'].append(model)
            else:
                fieldsnotinDfields.append([key+(20-len(key)*' '), model])

    print(f'{len(dparam_sub)} fields in the library \n')

    # ------------------
    # get column headers
    col2 = [
        'Field', 'definition', 'group', 'value', 'units', 'In model'

    ]

    # ------------------
    # get values
    ar2 = [
        [k0,
         v0['definition'],
         v0['group'],
         v0['value'],
         v0['units'],
         str(v0['inmodel'])
         ]
        for k0, v0 in dparam_sub.items() if v0['group'] != 'Numerical'
    ]

    return _utils._get_summary(
        lar=[ar2],
        lcol=[col2],
        verb=True,
        returnas=returnas,
    )


def create_preset_from_model_preset(targetmodel,
                                    outputmodel,
                                    targetpreset=False,
                                    targetdpreset=False,
                                    returnas='hub'):
    '''
    Open targetmodel, with or without preset/dpreset, and then gives all necessary
    values to outputmodel so that, if they solve the same equations on different approaches,
    they give the same result

    Please note that if they have different mechanism inside with different
    parameters, you have to manually set them so that they have the same behavior.


    Parameters
    ----------
    targetmodel : model name of the model we will copy the value
        DESCRIPTION.
    targetpreset : the preset name for targetmodel. Optional
        DESCRIPTION. The default is False.
    targetdpreset : the dictionnary preset for targetpreset
        DESCRIPTION. The default is False.
    outputmodel : the name of the model we will use after
        DESCRIPTION.
    returnas : (dict,'hub','dpreset') gives different type of output depending of the situation :
        * dict is a dict of field : value
        * hub is outputmodel loaded with the preset
        * dpreset is a dictionnary with this preset inside
        The default is 'hub'.

    Returns
    -------
    None.

    '''
    # LOADING TARGET
    # IF PRESET AND PRESET FILE GIVEN
    if targetpreset and targetdpreset:
        hub = Hub(targetmodel, preset=targetpreset,
                  dpresets=targetdpreset, verb=False)

    # ELIF PRESET NAME GIVEN
    elif targetpreset:
        hub = Hub(targetmodel, preset=targetpreset, verb=False)

    # ELSE USE OF BASIC VALUES
    else:
        hub = Hub(targetmodel, verb=False)

    hub_output = Hub(outputmodel, verb=False)

    # COPY OF THE PARAMETERS INTO A NEW DICTIONNARY
    FieldToLoad = hub_output.get_dparam(returnas=dict, eqtype=[None, 'ode'])
    # group=('Numerical',),)
    R = hub.get_dparam(returnas=dict)
    tdic = {}
    for k, v in FieldToLoad.items():
        val = R[k]['value']
        if 'initial' in v.keys():
            tdic[k] = val[0][0]
        else:
            tdic[k] = val
    _DPRESETS = {'Copy'+targetmodel: {'fields': tdic, }, }

    if returnas == dict:
        return tdic
    if returnas == 'hub':
        return Hub(outputmodel, preset='Copy'+targetmodel,
                   dpresets=_DPRESETS, verb=False)
    if returnas == 'preset':
        return _DPRESETS


def plot_one_run_all_solvers(_MODEL, preset=False, _DPRESET=False):
    '''
    Will compare up to seven solver that exist in pygemmes, using the model and preset you provide
    * if no preset, nor dictionary of preset the system takes default values
    * if preset and not dictionary preset, preset must be the name of one of the model preset
    * if preset and _dpreset, the system will load the preset in _dpreset

    Parameters
    ----------
    _MODEL : TYPE
        Name of the model for test
    preset : TYPE, optional
        Name of the preset. if none default value
    _DPRESET : TYPE, optional
        preset dictionary. if none/false, presets from the model

    Returns
    -------
    Print of all solvers super-imposed
    '''

    colors = ['y', 'k', 'm', 'r', 'g', 'b', 'c']
    dsolvers = get_available_solvers(returnas=dict, verb=False,)

    # key is solver name, value is hub
    dhub = {}
    for ii, solver in enumerate(dsolvers):
        print('Solver :', solver)

        # LOADING OF THE MODEL WITH THE CORRESPONDING PRESET
        # IF PRESET AND PRESET FILE GIVEN
        if preset and _DPRESET:
            dhub[solver] = Hub(_MODEL, preset=preset,
                               dpresets=_DPRESET, verb=False)

        # ELIF PRESET NAME GIVEN
        elif preset:
            dhub[solver] = Hub(_MODEL, preset=preset, verb=False)

        # ELSE USE OF BASIC VALUES
        else:
            dhub[solver] = Hub(_MODEL, verb=False)

        # RUN
        dhub[solver].run(verb=1.1, solver=solver)

        # PRINT
        # If first solver, creation of dax
        if ii == 0:
            dax = dhub[solver].plot(
                label=solver, color=colors[ii],
                wintit='Solver comparison on model'+_MODEL,
                tit='Solver comparison on model'+_MODEL)
        # Else use of dax
        else:
            dax = dhub[solver].plot(label=solver,
                                    color=colors[ii], dax=dax)


def comparesolver_Lorenz(dt=0.01, Npoints=False):
    '''
    for a given timestep (default dt=0.01), compute a Lorenz attractor with
    each solver to compare them
    '''
    Coor = {}
    _MODEL = 'LorenzSystem'
    dmodels = get_available_models(returnas=dict, details=False, verb=False,)
    for solver in get_available_solvers(returnas=dict, verb=False,).keys():
        for preset in dmodels[_MODEL]['presets']:
            hub = Hub(_MODEL, preset=preset, verb=False)
            hub.set_dparam(key='dt', value=dt, verb=False)
            hub.run(verb=0, solver=solver)
            if Npoints:
                hub.reinterpolate_dparam(Npoints)
            R = hub.get_dparam(returnas=dict)
            Coor[solver] = {'x': R['x']['value'],
                            'y': R['y']['value'],
                            'z': R['z']['value'], }

    fig = plt.figure('', figsize=(10, 10))
    ax = plt.axes(projection='3d')
    for solver in get_available_solvers(returnas=dict, verb=False,).keys():
        ax.plot(Coor[solver]['x'][:, 0],
                Coor[solver]['y'][:, 0],
                Coor[solver]['z'][:, 0], label=solver, lw=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    plt.legend()
    plt.show()


def testConvergence_DampOsc(vecdt, solver, returnas='plot', getsummary=True):
    '''
    This test compute for multiple timestep the difference between the
    computed solution and the theoretical one on a damped oscillator.
    The output is either a plot (mean error in function of timestep) or
    a dictionnary associating each values
    '''

    import numpy as np
    Error = {}

    for dt in vecdt:
        print("dt:", dt)
        _DPRESETS = {'Convergence test':
                     {'fields': {'Tmax': 20,
                                 'dt': dt,
                                 'gamma': 1,
                                 'Omega': 10,
                                 'Final': 0,
                                 'theta': 00,
                                 'thetap': 1,
                                 },
                      },
                     }
        # Load and preset
        hub = Hub('DampOscillator', preset='Convergence test',
                  dpresets=_DPRESETS, verb=False)
        hub.run(verb=0, solver=solver)

        # Get the informations right
        R = hub.get_dparam(returnas=dict)
        gamma = R['gamma']['value']
        Omega = R['Omega']['value']
        Final = R['Final']['value']
        theta0 = R['theta']['value'][0]
        thetap0 = R['thetap']['value'][0]
        t = R['time']['value']
        thetanum = R['theta']['value']

        # Comparing strategies
        thetaTheo = (1/Omega)*np.real(thetap0*np.exp(-gamma*(t)) *
                                      np.sin(np.sqrt(Omega**2-gamma**2)*(t)))
        Error[dt] = np.mean(np.sqrt((thetanum-thetaTheo)**2))

    ################################
    if getsummary:
        hub.get_summary()

    dtlist = Error.keys()
    errorlist = [Error[k] for k in Error.keys()]

    Z = sorted(zip(dtlist, errorlist))

    dtlist = [x for x, y in Z]
    errorlist = [y for x, y in Z]

    if returnas == 'plot':
        plt.figure('Convergence')
        plt.subplot(211)
        plt.loglog(dtlist, errorlist, '-*')
        plt.axis('scaled')
        plt.ylabel('mean error')
        plt.xlabel('dt')
        plt.suptitle('Convergence test for'+solver)
        plt.subplot(212)
        ax2 = plt.gca()
        ax22 = plt.twinx(ax2)
        plt.plot(t, thetaTheo, label='Theory')
        plt.plot(t, thetanum, label='Numerical')
        ax22.plot(t, (thetanum-thetaTheo), ls='--', c='r')
        plt.legend()
        plt.show()
    else:
        return Error


def GenerateIndividualSensitivity(key, mu, sigma, disttype='normal', dictpreset={}, N=10):
    '''
    Generate a preset taking random values in one distribution.

    INPUT :
        * key : the field name you want to test the sensitivity
        * mu : the first parameter of your distribution (mean typically)
        * sigma : the second parameter of your distribution (std typically)
        * dispreset : dictionnary you want to add the distribution in
        * disttype : the type of distribution you pick the value on :
            1. 'log','lognormal','log-normal' for lognormal distribution
            2. 'normal','gaussian' for gaussian distribution
        * N : the number of value you want to pick

        IF THE DISTRIBUTION IS LOG, then mu is the median value

    '''

    if disttype in ['log', 'lognormal', 'log-normal']:
        dictpreset[key] = np.random.lognormal(np.log(mu), sigma, N)
    elif disttype in ['normal', 'gaussian']:
        dictpreset[key] = np.random.normal(mu, sigma, N)
    else:
        raise Exception('wrong distribution type input')
    return dictpreset


def GenerateCoupledSensitivity(InputDic, dictpreset={}, N=10, grid=False):
    '''
    Wrapup around GenerateIndividualSensitivity function, to generate multiple distributions entangled.

    InputDic should look like :
        {
        'alpha': {'mu': .02,
                  'sigma': .2,
                  'type': 'log'},
        'k2': {'mu': 20,
               'sigma': .2,
               'type': 'log'},
        'mu': {'mu': 1.3,
               'sigma': .2,
               'type': 'log'},
        }

    Be careful, grid will generate N**len(InputDic.key()) run if activated !

    '''
    for key, val in InputDic.items():
        dictpreset = GenerateIndividualSensitivity(key,
                                                   val['mu'],
                                                   val['sigma'],
                                                   disttype=val['type'],
                                                   dictpreset=dictpreset,
                                                   N=N)

    if grid:
        for key, val in dictpreset.items():
            dictpreset[key] = {'value': val, 'grid': True}
    return dictpreset


def Generate_network_logics(_MODEL,
                            screensize=1080,
                            custom=False,
                            smoothtype='dynamic'):
    '''
    Generate an HTML file showing you interactively how are variables linked with their adequate units

    Parameters
    ----------
    _MODEL : Model name you want to show
    screensize : TYPE, optional
        DESCRIPTION. The default is 1080.
    custom : TYPE, optional
        DESCRIPTION. The default is False.
    smoothtype :    Possible options: 'dynamic', 'continuous',
                    'discrete', 'diagonalCross', 'straightCross',
                    'horizontal', 'vertical', 'curvedCW',
                    'curvedCCW', 'cubicBezier'.
                    When using dynamic, the edges will have an
                    invisible support node guiding the shape.
                    This node is part of the physics simulation.
                    Default is set to continous.

    Returns
    -------
    None.

    '''
    from pyvis.network import Network
    hub = Hub(_MODEL, verb=False)
    R = hub.get_dparam(returnas=dict)

    ODENodes = hub.dfunc_order['ode']
    ODENodes.remove('time')
    StatevarNodes = hub.dfunc_order['statevar']

    net = Network(directed=True, height=screensize, width=screensize,
                  heading=_MODEL+' Logical network')

    for key in ODENodes:
        v = R[key]
        Title = ''
        Title += 'Units        :' + v['units']+'<br>'
        Title += 'Equation     :'+f'd{key}/dt=' + v['source_exp'].replace(
            'itself', key).replace('lamb', 'lambda')+'<br>'
        Title += 'definition   :' + v['definition']+'<br>'
        Title += 'Comment      :' + v['com']+'<br>'
        Title += 'Dependencies :'+'<br>'
        for key2 in [v2 for v2 in v['kargs'] if v2 != 'itself']:
            v1 = hub.dparam[key2]
            Title += '    '+key2 + (8-len(key2))*' ' + \
                v1['units']+(8-len(v1['units']))*' '+v1['definition']+'<br>'

        net.add_node(key,
                     label=key,  # R[key]['symbol'],
                     color=['#3da831'],
                     title=Title.replace(' ', '&nbsp;'),
                     group='ODE',
                     level=1,
                     shape='ellipse')

    for key in StatevarNodes:

        v = R[key]
        Title = ''
        Title += 'Units        :' + v['units']+'<br>'
        Title += 'Equation     :'+f'{key}=' + v['source_exp'].replace(
            'itself', key).replace('lamb', 'lambda')+'<br>'
        Title += 'definition   :' + v['definition']+'<br>'
        Title += 'Comment      :' + v['com']+'<br>'
        Title += 'Dependencies :'+'<br>'
        for key2 in [v2 for v2 in v['kargs'] if v2 != 'itself']:
            v1 = hub.dparam[key2]
            Title += '    '+key2 + (8-len(key2))*' ' + \
                v1['units']+(8-len(v1['units']))*' '+v1['definition']+'<br>'

        net.add_node(key,
                     label=key,  # R[key]['symbol'],
                     color=['#3da831'],
                     title=Title.replace(' ', '&nbsp;'),
                     group='STATEVAR',
                     level=2,
                     shape='ellipse')

    for k in ODENodes+StatevarNodes:
        v = R[k]
        for k2 in [k3 for k3 in v['kargs'] if k3 in ODENodes+StatevarNodes]:
            net.add_edge(k2, k)
        if 'itself' in v['kargs']:
            net.add_edge(k, k)

    net.set_edge_smooth('dynamic')

    net.repulsion(node_distance=100, spring_length=200)
    if custom:
        net.show_buttons(filter_=False)

    # net.prep_notebook()
    net.show(_MODEL+'edges.html')
