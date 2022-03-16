#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:24:17 2022

@author: matt
"""

import _core
import numpy as np
import matplotlib.pyplot as plt

hub = _core.Hub('Putty-Clay_carbon')
hub.run()
sol = hub.get_dparam(returnas=dict)

t = sol['time']['value']
N = sol['N']['value']
a = sol['a']['value']
w = sol['w']['value']
c_tax = sol['c_tax']['value']
GDP = sol['GDP']['value']
C = sol['C']['value']
id_kc_min = sol['id_kc_min']['value']
kc_opt = sol['kc_optimum']['value']
kc_ratios = sol['kc_ratios']['value']
carbon_intensity = sol['carbon_intensity']['value']

plt.scatter(GDP / (a * N), w / a + c_tax * C / GDP, c=t, s=1)


for index in np.linspace(0, t.size - 1, 20).astype(int):
    plt.loglog(kc_ratios, carbon_intensity[index].T)