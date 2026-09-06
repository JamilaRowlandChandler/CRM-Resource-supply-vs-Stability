# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:28:07 2026

@author: jamil

For the "Externally-supplied resources" model, sweep the shared
influx/outflux (chemostat turnover) rate down over many orders of magnitude
and track how the proportion of feasible and dynamically stable communities
changes - i.e. whether communities become chaotic as external resource
supply becomes vanishingly small relative to consumption.
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

repo_root = file_directory_name
while not os.path.isdir(os.path.join(repo_root, '.git')):
    repo_root = os.path.dirname(repo_root)

data_directory = os.path.join(repo_root, 'Data')

sys.path.insert(0, os.path.join(repo_root, 'external_resource_stability'))
from simulation_functions_new import CRM_across_parameter_space

sys.path.insert(0, os.path.join(repo_root, 'cavity_method_functions'))
from self_consistency_equation_functions import variable_fixed_parameters

# %%

def migration(migration_exponents,
              fixed_parameters,
              subdirectory = 'external_resource_stability/simulations/influx_outflux',
              save_method = 'v3',
              **kwargs):

    '''

    Simulate and save communities across a range of chemostat turnover
    rates (influx = outflux = 10**(-migration_exponents)), for a fixed
    model and the remaining (fixed) parameters.

    '''

    parameters = generate_parameters(migration_exponents,
                                     fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               ['exponent', 'exponent'],
                               save_method=save_method,
                               model = 'Externally-supplied resources',
                               **kwargs)
    
# %%

def generate_parameters(migration_exponents, fixed_parameters):

    '''

    Build one parameter dict per turnover rate, with influx (b) and outflux
    (o) both set to the same rate, 10**(-exponent) - so exponent = 0 gives a
    fast turnover rate of 1, and larger exponents give a vanishingly small
    (but still balanced) turnover rate. mu_c/mu_g/sigma_c/sigma_g are
    supplied in the M-independent cavity-method convention and rescaled
    here to per-resource means/std. devs (mu/M, sigma/sqrt(M)) - see
    rho_vs_sigma_es_vs_sl.py for the full rescaling rationale.

    '''

    migration_rates = 10**(-migration_exponents)

    fixed_parameters_mod = deepcopy(fixed_parameters)

    fixed_parameters_mod['mu_c'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['mu_g'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['sigma_c'] *= 1/np.sqrt(fixed_parameters_mod['M'])
    fixed_parameters_mod['sigma_g'] *= 1/np.sqrt(fixed_parameters_mod['M'])

    # array of all parameter combinations
    parameters = variable_fixed_parameters(np.vstack([migration_exponents,
                                                      migration_rates,
                                                      migration_rates]),
                                           fixed_parameters_mod,
                                           ['exponent', 'b', 'o'])

    return parameters

# %%

def load_clean_simulations(data_location):

    def prop_feasible(x,
                      feasibility_threshold = 1000):

        # "Divergence measure" is the simulation's final timepoint (see
        #   community_dynamics_df() in simulation_functions_new.py); a run
        #   that reached the full simulation end time (rather than being cut
        #   short by the unbounded_growth event) is treated as feasible
        return np.count_nonzero(x == feasibility_threshold)/len(x)


    def prop_stable(x,
                    stability_threshold = 0):

        # a community is "stable" here if its max. Lyapunov exponent < 0
        return np.count_nonzero(x < stability_threshold)/len(x)
    
    full_location = os.path.join(data_directory, 'external_resource_stability',
                                 'simulations', data_location)
    
    if full_location.endswith(".csv"):
    
        df = pd.read_csv(full_location, index_col=False)
            
    else: 
       
        df = pd.concat([pd.read_csv(full_location + "/" + file, index_col=False) 
                       for file in os.listdir(full_location)],
                       axis = 0, ignore_index = True) 
        
    df = df.apply(pd.to_numeric, errors="coerce")
    
    df.rename(columns = {"maxLe" : "Max. lyapunov exponent"}, inplace = True)
    df = np.round(df, 7)

    stable_feasible = df.groupby('b_val').agg({'Divergence measure' : prop_feasible,
                                               'Max. lyapunov exponent' : prop_stable}).reset_index().rename(columns = {'b_val' : 'influx',
                                                                                                                        'Divergence measure' : 'P(Feasible)',
                                                                                                                        'Max. lyapunov exponent' : 'P(Stable)'})
    
    return df, stable_feasible

# %%

# turnover rate = 10**(-exponent): swept from 1 (exponent 0) down to 1e-8
migration_exponents = np.arange(0, 8, 0.5)
mu = 50       # mean growth/consumption rate (M-independent scale)
sigma = 10.5  # heterogeneity in growth/consumption rates (M-independent scale)
rho = 0.4     # growth-consumption correlation

# %%

migration(migration_exponents,
          dict(mu_c = mu, sigma_c = sigma,
               mu_g = mu, sigma_g = sigma, rho = rho,
               d = 1, M = 150, S = 150),
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)
    
    