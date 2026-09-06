# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:24:01 2025

@author: jamil

Sweep growth-consumption correlation (rho) against consumption/growth rate
heterogeneity (sigma), comparing how the (in)stability transition differs
between the "Externally-supplied resources", "Self-limiting resource
supply" and "Hybrid resource supply" models (all at resource pool size
M = 150). Also generates a handful of single-community example simulations
(used elsewhere for illustrative time-series plots) and some hybrid-model
runs that vary how much self-inhibition (`a`) vs. outflux (`o`) contributes
to resource regulation.
"""

import numpy as np
import sys
import os
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, file_directory_name.removesuffix("\\stability_transitions"))
from simulation_functions_new import CRM_across_parameter_space, le_pivot_r

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\cavity_method_functions")
from self_consistency_equation_functions import variable_fixed_parameters, \
    parameter_combinations

# %%

def rho_sigma(model,
              rho_range,
              sigma_range,
              fixed_parameters,
              subdirectory,
              save_method = 'v3',
              **kwargs):

    '''

    Simulate and save communities across every combination of rho and sigma,
    for a fixed system size, model, and the remaining (fixed) parameters.

    '''

    parameters = generate_parameters(rho_range, sigma_range, fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               ['rho', 'sigma_M'],
                               save_method=save_method,
                               model = model,
                               **kwargs)
                    
# %%

def generate_parameters(rho_range, sigma_range, fixed_parameters):

    '''

    Build one parameter dict per (rho, sigma) combination, converting the
    "raw" (M-independent) rho and sigma values into the actual per-rate
    means/std. devs. that growth_consumption_rates() expects (see the
    matching function in es_largeM.py for the full rescaling rationale:
    mu_c, mu_g -> mu/M; sigma -> sigma/sqrt(M); rho is left unchanged).

    '''

    # sigma_range is labelled "sigma_M" below, since it's the sigma value
    #   in the M-independent parameterisation (before the /sqrt(M) rescaling)
    rho_sigma_combos = np.unique(parameter_combinations([rho_range,
                                                         sigma_range],
                                                        1),
                                    axis = 1)

    # sigma_c and sigma_g share the same value here (growth and consumption
    #   rates are drawn with equal heterogeneity)
    variable_parameters = np.vstack([rho_sigma_combos,
                                     rho_sigma_combos[1, :]/np.sqrt(fixed_parameters['M']),
                                     rho_sigma_combos[1, :]/np.sqrt(fixed_parameters['M'])])

    fixed_parameters_mod = deepcopy(fixed_parameters)

    fixed_parameters_mod['mu_c'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['mu_g'] *= 1/fixed_parameters_mod['M']

    # array of all parameter combinations
    parameters = variable_fixed_parameters(variable_parameters,
                                           fixed_parameters_mod,
                                           ['rho', 'sigma_M',
                                            'sigma_c', 'sigma_g'])

    return parameters



# %%

# rho: growth-consumption correlation, swept from uncorrelated (0.1) to
#   perfectly correlated (1.0)
rhos = np.arange(0.1, 1.1, 0.1)
# sigma: heterogeneity in growth/consumption rates (M-independent scale)
sigmas = np.arange(2, 13, 1)
mu = 50    # mean growth/consumption rate (M-independent scale)
d = 1      # consumer death rate
b = 1      # resource influx/intrinsic growth rate
o = 1      # resource outflux (dilution) rate
a = 1      # resource self-inhibition (quadratic) coefficient
system_size = 150

# %%

rho_sigma("Externally-supplied resources",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b, o = o,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_es",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

rho_sigma("Self-limiting resource supply",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_sl",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# %%

# Example simulations
#
# Both use the "Hybrid resource supply" model (dRdt = B + O*R - A*R^2 - consumption),
# with (b, o, a) chosen so its dynamics reduce to the ES or SL special case -
# this keeps both example communities on the same model class/save format
# for the downstream example time-series figures.

# o = -o, a = 0  ->  dRdt = b - o*R - consumption : matches "Externally-supplied resources"
rho_sigma("Hybrid resource supply",
          [0.2, 1.0],
          [10.0],
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b, o = -o, a = 0,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_es_examplesim",
          no_communities = 1,
          t_end = 1000,
          no_init_conds = 1,
          save_method="v1")

# b = 0, o = 1, a = 1  ->  dRdt = R*(1 - R) - consumption : matches "Self-limiting resource supply"
rho_sigma("Hybrid resource supply",
          [0.2, 0.9, 1.0],
          [6.0],
          dict(mu_c = mu, mu_g = mu,
               d = d, b = 0, o = o, a = 1,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_sl_examplesim",
          no_communities = 1,
          t_end = 1000,
          no_init_conds = 1,
          save_method="v1")

# %%

# Full rho-sigma sweeps of the Hybrid model, to see how mixing resource
# self-inhibition (a) and outflux (o) - rather than using either alone as
# in the ES/SL special cases above - changes the stability transition.

# non-negligable self-inhibition (a = 1, comparable to o = 1)
rho_sigma("Hybrid resource supply",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b, o = o, a = a,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_h",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# as above, but with a much smaller resource influx (b)
rho_sigma("Hybrid resource supply",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = 0.001, o = o, a = a,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_h_b0001",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# negligable self-inhibition (a ~ 0, so outflux o dominates resource regulation)
rho_sigma("Hybrid resource supply",
          np.arange(0.7, 1.0, 0.1), #rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = 0.001, o = o, a = 10**(-5),
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_smallinflux",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# as above (negligable self-inhibition), but with the normal-scale influx (b = 1)
rho_sigma("Hybrid resource supply",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b, o = o, a = 10**(-5),
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_largeinflux",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# %%

################################################

#rho_sigma("Hybrid resource supply",
#          [0.6], # np.arange(0.7, 1.0, 0.1), #rhos,
#          np.arange(5.0, 13.0, 1.0), #sigmas,
#          dict(mu_c = mu, mu_g = mu,
#               d = d, b = 0, o = o, a = 10**(-5),
#               M = system_size, S = system_size),
#          "external_resource_stability/simulations/rho_sigma_noinflux",
#          no_communities = 20,
#          t_end = 1000,
#          no_init_conds = 1)