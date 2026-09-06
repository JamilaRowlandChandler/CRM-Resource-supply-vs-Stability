# -*- coding: utf-8 -*-
"""
Created on Sun May  4 13:01:05 2025

@author: jamil
"""

import numpy as np
import numpy.typing as npt
from typing import Literal, Union, TypedDict
from scipy.integrate import solve_ivp

from parameters import ParametersInterface
from differential_equations import DifferentialEquationsInterface, unbounded_growth #, ReloadedODEs
from community_level_properties import CommunityPropertiesInterface

# %%

def Consumer_Resource_Model(model : Literal["Self-limiting resource supply",
                                            "Self-limiting resource supply, leached",
                                            "Self-limiting resource supply, self-inhibition",
                                            "Self-limiting resource supply, multi-trophic level"
                                            "Externally-supplied resources"],
                            no_species : Union[int, None] = None,
                            no_resources : Union[int, None] = None,
                            trophic_levels : Union[int, None] = None,
                            pool_sizes : Union[int, None] = None):
    
    '''
    
    Wrapper for different consumer resource model classes

    Parameters
    ----------
    model : str
        Type of consumer resource model. Options are:
            "Self-limiting resource supply" - resources grow logistically
            "Self-limiting resource supply, self-inhibition" - same as 
            "Self-limiting resource supply", but with direct consumer self-inhibition
            "Externally-supplied resources" - chemostat-style resource dynamics
            (constant influx + dilution)
    no_species : int
        species pool size
    no_resources : int
        resource pool size

    Raises
    ------
    Exception
        If a non-existent model is selected.

    Returns
    -------
    instance : object of some consumer-resource model class
        Instance of some consumer-resource model class.

    '''
    
    match model:
        
        case "Self-limiting resource supply":
            
            instance = SL_CRM(no_species, no_resources)
            
        case "Externally-supplied resources":
            
            instance = ES_CRM(no_species, no_resources)
            
        case "Hybrid resource supply":
            
            instance = Hybrid_CRM(no_species, no_resources)
            
        case _:
            
            raise Exception('You have not selected an exisiting model.\n' + \
                  'Please chose from either "Self-limiting resource supply"' + \
                      '"Externally-supplied resources"' + \
                      ' or "Hybrid resource supply"')
    return instance

# %%

class SL_CRM(ParametersInterface,
             DifferentialEquationsInterface,
             #ReloadedODEs,
             CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply
    
    '''
    
    def __init__(self, no_species : int, no_resources : int):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        no_species : int
            species pool size
        no_resources : int
            resource pool size

        Returns
        -------
        None.

        '''
        
        # assign species and resource pool size as class attributes
        self.no_species = no_species
        self.no_resources = no_resources
        
    def model_specific_rates(self,
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             resource_growth_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             resource_growth_args : 
                                 Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                       TypedDict('constant', {'b' : float}),
                                       TypedDict('user-supplied', {'b' : npt.NDArray})]
                                 = {'b' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        resource_growth_method : str
            Method used to generate intrinsic resource growth rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        resource_growth_args : dict
            Arguments for resource_growth_method. Options are the same as 
            death_method args.

        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, resource_growth_method],
                [death_args, resource_growth_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
    
    #####################################################################
    
    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  S, G, C, D, B):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            S : int
                species pool size (used to separate y into species and resource 
                                   abundances)
            G : np.ndarray
                matrix of consumer growth rates
            C : np.ndarray
                matrix of resource consumption rates
            D : np.ndarray
                consumer death rates
            B : np.ndarray
                intrinsic resource growth rates

            Returns
            -------
            np.ndarray
                Rate of change in species and resource abundances over time 
                (dNdt and dRdt)

            '''
            
            # extinction threshold
            #y[y < 1e-5] = 0
            
            # separate species and resource abundances
            species, resources = y[:S], y[S:]
            
            # change in consumer abundances over time
            dNdt = species * (np.sum(G * resources, axis = 1) - D)
            
            # change in resource abundances over time
            dRdt = (resources * (B - resources)) - \
                (resources * np.sum(C * species, axis=1))
                
            return np.concatenate((dNdt, dRdt)) + 1e-8

        def jacobian(t, y,
                    S, G, C, D, B):

            '''

            Analytical Jacobian for the CRM with self-limiting resource supply.

            J[:S, :S]  = diag(G.R - D)
            J[:S, S:]  = G * N        (d(dN/dt)/dR)
            J[S:, :S]  = -C * R       (d(dR/dt)/dN)
            J[S:, S:]  = diag(B - 2R - C.N)

            '''

            species, resources = y[:S], y[S:]

            growth_term = np.sum(G * resources, axis = 1) - D
            consumption_term = np.sum(C * species, axis = 1)

            J = np.zeros((y.size, y.size))

            J[:S, :S] = np.diag(growth_term)
            J[:S, S:] = G * species[:, np.newaxis]
            J[S:, :S] = -C * resources[:, np.newaxis]
            J[S:, S:] = np.diag(B - 2*resources - consumption_term)

            return J

        unbounded_growth.terminal = True

        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)
        return solve_ivp(model, [0, t_end], initial_abundance,
                         args = (self.no_species, self.growth, self.consumption,
                                 self.d, self.b),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200),
                         jac = jacobian,
                         events = unbounded_growth)

# %%

class ES_CRM(ParametersInterface, DifferentialEquationsInterface,
             CommunityPropertiesInterface):
    
    def __init__(self, no_species, no_resources):
        
        self.no_species = no_species
        self.no_resources = no_resources
    
    def model_specific_rates(self, 
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             influx_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             influx_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'b' : float}),
                                                TypedDict('user-supplied', {'b' : npt.NDArray})]
                             = {'b' : 1},
                             outflux_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             outflux_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'o' : float}),
                                                TypedDict('user-supplied', {'o' : npt.NDArray})]
                             = {'o' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        influx_method : str
            Method used to generate intrinsic resource influx rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        influx_args : dict
            Arguments for influx_method. Options are the same as 
            death_method args.
        outflux_method : str
            Method used to generate intrinsic resource outflux rates. Options are
            the same as death_method, but named 'o' rather than 'd'.
        outflux_args : dict
            Arguments for outflux_method. Options are the same as 
            death_method args.
        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b', 'o']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, ),
                     (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, influx_method, outflux_method],
                [death_args, influx_args, outflux_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
                
    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  S, G, C, D, B, O):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            S : int
                species pool size (used to separate y into species and resource 
                                   abundances)
            G : np.ndarray
                matrix of consumer growth rates
            C : np.ndarray
                matrix of resource consumption rates
            D : np.ndarray
                consumer death rates
            B : np.ndarray
                intrinsic resource growth rates

            Returns
            -------
            np.ndarray
                Rate of change in species and resource abundances over time 
                (dNdt and dRdt)

            '''
            
            # extinction threshold
            #y[y < 1e-5] = 0
            
            # separate species and resource abundances
            species, resources = y[:S], y[S:]
            
            # change in consumer abundances over time
            dNdt = species * (np.sum(G * resources, axis = 1) - D)
        
            # change in resource abundances over time
            dRdt = (B - O * resources) - \
                (resources * np.sum(C * species, axis=1))
                
            return np.concatenate((dNdt, dRdt)) + 1e-8

        def jacobian(t, y,
                    S, G, C, D, B, O):

            '''

            Analytical Jacobian for the CRM with externally-supplied resources.

            J[:S, :S]  = diag(G.R - D)
            J[:S, S:]  = G * N        (d(dN/dt)/dR)
            J[S:, :S]  = -C * R       (d(dR/dt)/dN)
            J[S:, S:]  = diag(-O - C.N)

            '''

            species, resources = y[:S], y[S:]

            growth_term = np.sum(G * resources, axis = 1) - D
            consumption_term = np.sum(C * species, axis = 1)

            J = np.zeros((y.size, y.size))

            J[:S, :S] = np.diag(growth_term)
            J[:S, S:] = G * species[:, np.newaxis]
            J[S:, :S] = -C * resources[:, np.newaxis]
            J[S:, S:] = np.diag(-O - consumption_term)

            return J

        unbounded_growth.terminal = True

        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)
        return solve_ivp(model, [0, t_end], initial_abundance,
                         args = (self.no_species, self.growth, self.consumption,
                                 self.d, self.b, self.o),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200),
                         jac = jacobian,
                         events = unbounded_growth)

# %%

class Hybrid_CRM(ParametersInterface, DifferentialEquationsInterface,
                 CommunityPropertiesInterface):
    
    def __init__(self, no_species, no_resources):
        
        self.no_species = no_species
        self.no_resources = no_resources
    
    def model_specific_rates(self, 
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             influx_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             influx_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'b' : float}),
                                                TypedDict('user-supplied', {'b' : npt.NDArray})]
                             = {'b' : 1},
                             outflux_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             outflux_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'o' : float}),
                                                TypedDict('user-supplied', {'o' : npt.NDArray})]
                             = {'o' : 1},
                             resource_inhibition_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             resource_inhibition_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                              TypedDict('constant', {'a' : float}),
                                                              TypedDict('user-supplied', {'a' : npt.NDArray})]
                             = {'a' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        influx_method : str
            Method used to generate intrinsic resource influx rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        influx_args : dict
            Arguments for influx_method. Options are the same as 
            death_method args.
        outflux_method : str
            Method used to generate intrinsic resource outflux rates. Options are
            the same as death_method, but named 'o' rather than 'd'.
        outflux_args : dict
            Arguments for outflux_method. Options are the same as 
            death_method args.
        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b', 'o', 'a']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, ),
                     (self.no_resources, ), (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, influx_method, outflux_method, resource_inhibition_method],
                [death_args, influx_args, outflux_args, resource_inhibition_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
                
    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  S, G, C, D, B, O, A):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            S : int
                species pool size (used to separate y into species and resource 
                                   abundances)
            G : np.ndarray
                matrix of consumer growth rates
            C : np.ndarray
                matrix of resource consumption rates
            D : np.ndarray
                consumer death rates
            B : np.ndarray
                intrinsic resource growth rates

            Returns
            -------
            np.ndarray
                Rate of change in species and resource abundances over time 
                (dNdt and dRdt)

            '''
            
            # extinction threshold
            #y[y < 1e-5] = 0
            
            # separate species and resource abundances
            species, resources = y[:S], y[S:]
            
            # change in consumer abundances over time
            dNdt = species * (np.sum(G * resources, axis = 1) - D)
            
            # change in resource abundances over time
            dRdt = (B + O * resources - A * resources**2) - \
                (resources * np.sum(C * species, axis=1))
                
            return np.concatenate((dNdt, dRdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance, 
                         args = (self.no_species, self.growth, self.consumption, 
                                 self.d, self.b, self.o, self.a),
                         method = 'LSODA', # 'Radau', #'RK45',
                         rtol = 1e-9, atol = 1e-7,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
