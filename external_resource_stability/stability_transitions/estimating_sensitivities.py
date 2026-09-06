# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:11:59 2026

@author: jamil
"""

import numpy as np
import sys
import os
from matplotlib import pyplot as plt 
import pandas as pd 
import seaborn as sns
from scipy.signal import find_peaks
from tqdm import tqdm

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

repo_root = file_directory_name
while not os.path.isdir(os.path.join(repo_root, '.git')):
    repo_root = os.path.dirname(repo_root)

sys.path.insert(0, os.path.join(repo_root, 'consumer_resource_modules'))
from models import Consumer_Resource_Model
from community_level_properties import max_le

sys.path.insert(0, os.path.join(repo_root, 'cavity_method_functions'))
from self_consistency_equation_functions import parameter_combinations

sys.path.insert(0, os.path.join(repo_root, 'external_resource_stability'))
from simulation_functions_new import pickle_dump

# %%

def sensitivity_influx_rho(influxes,
                           rho_sets,
                           no_communities = 20):
    
    all_combinations = np.hstack([parameter_combinations([rhos,
                                                         [influx]],
                                                         1)
                                  for influx, rhos in zip(influxes,
                                                          rho_sets)]).T.tolist()
    
    sensitivity_df = pd.DataFrame([community_sensitivity(rho, influx)
                                   for rho, influx in tqdm(all_combinations,
                                                           position = 0,
                                                           leave = True)
                                   for i in tqdm(range(no_communities),
                                                 position = 1,
                                                 leave = False)])
    
    return sensitivity_df

# %%

def example_sensitivities(rhos,
                          filename):
    
    influxes = 10**np.arange(-5.0, -1.5, 0.5)

    example_sensitivities = sensitivities_all_data(influxes,
                                                   rhos,
                                                   no_communities=80)

    pickle_dump(os.path.join(repo_root, 'Data', 'external_resource_stability',
                             'simulations', filename + ".pkl"),
                example_sensitivities)

# %%

def sensitivities_all_data(influxes,
                           rhos, 
                           no_communities = 60):
    
    return [community_sensitivity(rho, influx, all_data=True)
                                   for rho, influx in tqdm(zip(rhos, influxes),
                                                           position = 0,
                                                           leave = True)
                                   for i in tqdm(range(no_communities),
                                                 position = 1,
                                                 leave = False)]

# %%

def community_sensitivity(rho,
                          influx,
                          all_data = False):
    
    system_size = 150
    mu = 50
    sigma = 4
    
    crm_community = Consumer_Resource_Model("Hybrid resource supply",
                                            system_size,
                                            system_size)

    # generate model parameters
    crm_community.growth_consumption_rates(method = 'coupled by rho',
                                           mu_c = mu/system_size,
                                           sigma_c = sigma/np.sqrt(system_size),
                                           mu_g = mu/system_size,
                                           sigma_g = sigma/np.sqrt(system_size),
                                           rho = rho)

    crm_community.model_specific_rates(influx_method='constant',
                                       influx_args={'b' : influx})
    
    crm_community.simulate_community(1000, 1)
    crm_community.calculate_community_properties(sensitivity_distribution=True)
    crm_community.lyapunov_exponent = max_le(crm_community,
                                             crm_community.ODE_sols[0].y[:, -1],
                                             T = 1000,
                                             perturbation = 1e-6)
    
    dRdx = crm_community.consumption_sensitivity[0]['dRdx2']
    x = crm_community.consumption_sensitivity[0]['x']
    
    peaks, _ = find_peaks(dRdx[np.where((x > 0.7) & (x < 1.2))],
                          threshold = 0.01)
    
    if peaks.size > 0:
            
        max_peak = np.max(dRdx[peaks])
        
    else:
        
        max_peak = 0
        
        
    if all_data == True:
    
        return {'b' : influx,
                'rho' : rho,
                'x' : crm_community.consumption_sensitivity[0]['x'],
                'R' : crm_community.consumption_sensitivity[0]['R'],
                'dRdx2' : dRdx,
                'max peak' : max_peak,
                'max. le' : crm_community.lyapunov_exponent}
    
    else:
        
        return {'b' : influx,
                'rho' : rho,
                'dRdx2' : max_peak,
                'max. le' : crm_community.lyapunov_exponent}
    
# %%

influxes = 10**np.arange(-5.0, -1.5, 0.5)
max_rhos = np.arange(1.0, 0.70, -0.05)
decrease = np.arange(0, (len(influxes) - 1)*0.05, 0.05)

rho_sets = np.round(np.tile(max_rhos,
                             len(influxes)).reshape((len(influxes),
                                                     len(max_rhos))) 
                                                     - decrease[:, None],
                                                     6).tolist()

rho_influx_df = sensitivity_influx_rho(influxes,
                                       rho_sets) 

rho_influx_df['b'] = np.round(np.log10(rho_influx_df['b']), 6)

rho_influx_df['log_dRdx2'] = np.log10(np.select([rho_influx_df['dRdx2'] == 0,
                                                 rho_influx_df['dRdx2'] > 0],
                                                [np.nan,
                                                 rho_influx_df['dRdx2']],
                                                0))

rho_influx_df.to_csv(os.path.join(repo_root, 'Data', 'external_resource_stability',
                                  'simulations', 'rho_influx_sensitivities.csv'))

# %%

example_sensitivities(np.array([0.9, 0.875, 0.85, 0.8, 0.725, 0.6, 0.5]), #np.array([0.9, 0.9, 0.9, 0.85, 0.75, 0.7, 0.55]),
                      "rho_influx_example_sensitivities_cusp_2")
