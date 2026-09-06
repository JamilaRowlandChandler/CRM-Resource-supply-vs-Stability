# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:46:34 2026

@author: jamil

Sweep growth-consumption correlation (rho) against consumption/growth rate
heterogeneity (sigma) for the "Externally-supplied resources" CRM, at a
large resource pool size (M = 500). This is a large-M companion to the
rho vs. sigma sweep in rho_vs_sigma_es_vs_sl.py (run there at M = 150),
used to check that the rho-sigma (in)stability transition doesn't shift
with system size.
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
    for a fixed system size and the remaining (fixed) parameters.

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
    means/std. devs. that growth_consumption_rates() expects.

    fixed_parameters['mu_c']/['mu_g'] and sigma_range are supplied in the
    M-independent cavity-method convention (i.e. as if M = 1); to keep the
    community in the same statistical regime as smaller-M runs, they are
    rescaled here to mu_c/M, mu_g/M (mean consumption/growth per resource)
    and sigma/sqrt(M) (std. dev. in consumption/growth per resource).
    rho itself is a correlation coefficient, so it needs no rescaling.

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

# rho: growth-consumption correlation, swept from perfectly correlated (1.0)
#   down towards uncorrelated (0.3)
rhos = np.arange(1.0, 0.3, -0.1)
# sigma: heterogeneity in growth/consumption rates (M-independent scale)
sigmas = np.arange(2, 13, 1)
mu = 50    # mean growth/consumption rate (M-independent scale)
d = 1      # consumer death rate
b = 1      # resource influx rate
o = 1      # resource outflux (dilution) rate
system_size = 500

# %%

rho_sigma("Externally-supplied resources",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b, o = o,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_es_largeM",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)