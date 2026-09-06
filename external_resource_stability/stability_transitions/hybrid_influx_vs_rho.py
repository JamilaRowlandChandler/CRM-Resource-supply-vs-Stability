# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:52:11 2026

@author: jamil

Sweep resource influx (b, on a log scale) against growth-consumption
correlation (rho) for the "Hybrid resource supply" model, to see how the
(in)stability transition depends on how strongly resources are externally
supplied vs. correlated growth/consumption.
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
    
def p1_p2(model,
          p1,
          p2,
          fixed_parameters,
          subdirectory,
          save_method = 'v3',
          **kwargs):

    '''

    Simulate and save communities across every combination of two named
    parameters, p1 and p2 - each given as a (name, values) tuple, e.g.
    ('exponent_b', bs) or ('rho', rhos) - for a fixed model and the
    remaining (fixed) parameters. More general version of the rho/sigma
    sweep in rho_vs_sigma_es_vs_sl.py, allowing any pair of parameters
    (not just rho and sigma) to be varied.

    '''

    parameters = generate_parameters(p1,
                                     p2,
                                     fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               [p1[0], p2[0]],
                               save_method=save_method,
                               model = model,
                               **kwargs)
                        
# %%

def generate_parameters(p1,
                        p2,
                        fixed_parameters):

    def transform(param_array,
                  param_name,
                  fixed_parameters,
                  no_parm = 2):

        '''

        Convert a swept parameter into the value(s) growth_consumption_rates()/
        model_specific_rates() actually expect. p1/p2 names that aren't
        recognised here (e.g. 'rho', which needs no conversion) fall through
        to the default case and are used as-is.

        '''

        match param_name:

            case 'exponent_b':

                # b is swept on a log scale (as an exponent), so undo the log here
                return 10**param_array, ['b']

            case 'exponent_a':

                # a is swept on a log scale (as an exponent), so undo the log here
                return 10**param_array, ['a']

            case 'sigma':

                # rescale the M-independent sigma into sigma_c/sigma_g
                #   (std. dev. of consumption/growth rates), see
                #   rho_vs_sigma_es_vs_sl.py for the full rescaling rationale
                return np.tile(param_array/np.sqrt(fixed_parameters['M']),
                               2).reshape((len(param_array), no_parm)), \
                        ['sigma_c', 'sigma_g']

            case 'mu':

                # rescale the M-independent mu into mu_c/mu_g
                #   (mean consumption/growth rates)
                return np.tile(param_array/fixed_parameters['M'],
                               2).reshape((len(param_array), no_parm)), \
                        ['mu_c', 'mu_g']

            case _:

                # parameter needs no rescaling (e.g. 'rho'); nothing to add
                return [], []
                        
    p1_p2_combos = np.unique(parameter_combinations([p1[1],
                                                     p2[1]],
                                                    1),
                                    axis = 1).tolist()
    
    mod_params = []
    mod_names = []
    
    mod_params.append(np.array(p1_p2_combos))
    mod_names += [p1[0], p2[0]]
    
    for param_array, param_name in zip(p1_p2_combos, [p1[0], p2[0]]):
        
        params, names = transform(np.array(param_array),
                                  param_name,
                                  fixed_parameters)
        
        if names != []: 
            
            mod_params.append(params)
            mod_names += names
    
    variable_parameters = np.vstack(mod_params)
    
    fixed_parameters_mod = deepcopy(fixed_parameters)

    # if mu/sigma are fixed (rather than swept as p1/p2), they still need the
    #   same M-dependent rescaling as the transform() cases above
    if 'mu' in fixed_parameters_mod:

        fixed_parameters_mod['mu_c'] = fixed_parameters_mod['mu']/fixed_parameters_mod['M']
        fixed_parameters_mod['mu_g'] = fixed_parameters_mod['mu_c']

    if 'sigma' in fixed_parameters_mod:

        fixed_parameters_mod['sigma_c'] = fixed_parameters_mod['sigma']/np.sqrt(fixed_parameters_mod['M'])
        fixed_parameters_mod['sigma_g'] = fixed_parameters_mod['sigma_c']

    # array of all parameter combinations
    parameters = variable_fixed_parameters(variable_parameters,
                                           fixed_parameters_mod,
                                           mod_names)
    
    return parameters

#############################################################################

# %%

# resource influx b, swept as an exponent: 10**bs spans ~1e-5 to ~1 (log scale)
bs = np.arange(-5, 0.5, 0.5)
resource_inhibitions = np.array([-5, 0.0])  # (unused below)
# rho: growth-consumption correlation, swept from uncorrelated (0.1) to
#   perfectly correlated (1.0)
rhos = np.arange(0.1, 1.1, 0.1)
sigma = 4.0    # heterogeneity in growth/consumption rates (M-independent scale)
mu = 50        # mean growth/consumption rate (M-independent scale)
d = 1          # consumer death rate
o = 1          # resource outflux (dilution) rate
system_size = 150

# %%

p1_p2("Hybrid resource supply",
      ('exponent_b', bs),
      ('rho', rhos),
      dict(mu = mu, sigma = sigma,
           d = d,
           o = o,
           a = 1,
           M = system_size, S = system_size),
    "external_resource_stability/simulations/hybrid_influx_rho",
    no_communities = 20,
    t_end = 1000,
    no_init_conds = 1)

####################################################

