# -*- coding: utf-8 -*-
'''
Contains all the possibilities of each _core interaction
'''
# %% Test that the system is still doing great #############################
# !pytest tests/test_01_Hub.py -v

import _core
import _plots as plots

# _core._class_checks.models.get_available_models()
# _core._class_checks.models.describe_ALL_available_models()
# _core._class_checks.models.PrintDFIELDS()


hub = _core.Hub(model='TEMPLATE')
presetlist = hub.get_presets(returnas=list)

groupsofvariable = hub.get_groupsofvariable(returnas=dict)
# hub.set_preset('default')
# hub.Change_Attributes({'nu': 2.65,
#                       'K': 2})
hub.run(verb=1.1)


results = hub.get_dparam(returnas=dict)


hub.FillCyclesForAllVar(ref='lambda', idx='all')
plots.ForEachUnitsGroup(hub)
plots.Var(hub, 'lambda', idx=0, cycles=True, log=False)
plots.phasespace(hub, x='omega', y='d', idx=0)
plots.phasespace(hub, x='omega', y='lambda', color='time', idx=0)
plots.phasespace(hub, x='omega', y='lambda', color='d', idx=0)


plots.AllPhaseSpace(hub, groupsofvariable[''], idx=0)


plots.Var(hub, 'K', idx=0, cycles=True, log=True)
hub.save()
hub.get_summary()


sol_load = _core._saveload.load(' ')

for key, val in groupsofvariable.items():
    print(key, val)
