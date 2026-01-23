import openmc 
import numpy as np
from typing import List, Iterable
def create_openmc_settings(batches : int, samples : int, source : openmc.SourceBase, verbosity : int = 1, seed : int = None):
    '''
    Function to create OpenMC settings for a fixed source simulation
    Parameters
    ----------
    batches : int
        The number of batches to run
    samples : int
        The number of particles to simulate per batch
    source : openmc.SourceBase
        The OpenMC Source object for the simulation
    mg : bool
        If True, use multi-group mode (default is False)
    verbosity : int
        The verbosity level of the simulation (default is 1)
    seed : int or None
        The random seed for the simulation. If None, a random seed is generated (default is None)
    Returns
    -------
    settings : openmc.Settings
        The OpenMC Settings object for the simulation
    -------
    '''
    settings = openmc.Settings()
    settings.batches       = batches
    settings.particles     = int(samples)
    settings.inactive      = 0
    settings.verbosity     = verbosity

    if seed is not None:
        settings.seed          = seed
    else:
        settings.seed          = int(np.random.random(1)[0] * 1e13)    
    
    settings.run_mode      = 'fixed source'
    settings.source           = source
    settings.survival_biasing = False
    settings.output           = {'tallies' : False}    
    
    return settings


def create_mgxs_lib(openmc_geometry : openmc.Geometry, egroups : Iterable[float], legendre_order : int, extra_types  : List[str] = ['(n,Xt)', 'heating']):    
    '''
    Function to create a openmc.mgxs.Library for generating multigroup cross sections in the 1D breeding blanket simulation.
    
    Parameters
    ----------
    openmc_geometry : openmc.Geometry
        The OpenMC geometry object for the breeding blanket
    egroups : Iterable[float]
        The energy group boundaries for the multi-group cross-section library, ordered from low to high
    legendre_order : int
        The Legendre order for the multi-group cross-section library
    extra_types : List[str]
        A list of extra multi-group cross-section types to include in the library (default is ['(n,Xt)', 'heating'])
    Returns
    -------
    mgxs_lib : openmc.mgxs.Library
        The OpenMC multi-group cross-section library object for the breeding blanket
    -------
    
    '''
    
    mgxs_lib                = openmc.mgxs.Library(openmc_geometry)
    mgxs_lib.energy_groups  = openmc.mgxs.EnergyGroups(egroups) # from low to high??
    mgxs_lib.mgxs_types     = ['total', 'absorption', 'nu-fission', 'fission','nu-scatter matrix', 'multiplicity matrix', 'chi', 'scatter', 'nu-scatter', 'scatter matrix']  + extra_types
    mgxs_lib.domain_type    = "material"
    mgxs_lib.domain         = list(openmc_geometry.get_all_materials().values())
    mgxs_lib.correction     = None
    mgxs_lib.by_nuclide     = False
    mgxs_lib.legendre_order = legendre_order

    mgxs_lib.check_library_for_openmc_mgxs()
    mgxs_lib.build_library()
    return mgxs_lib


def mgxs_lib_to_standard_formatting(mgxs_lib : openmc.mgxs.Library, conversion_factor = 100.0):
    '''
    Function to convert an OpenMC mgxs.Library to a standard dictionary format for easier use
    
    This converts it to the following format:

    sigma_l_gout_gin = [l, g_out, g_in] 

    instead of the normal OpenMC format of

    [g_in, g_out, l]

    also applies a conversion factor


    Parameters
    ----------
    mgxs_lib : openmc.mgxs.Library
        The OpenMC multi-group cross-section library object for the breeding blanket
    conversion_factor : float
        A conversion factor to apply to the cross-section data (default is 100.0 to convert from cm^-1 to m^-1)

    Returns
    -------
    mgxs_dict : dict
        A dictionary containing the multi-group cross-section data in a standard format
    -------

    '''
    ignored_types = ['scatter matrix', 'multiplicity', 'chi']

    mgxs_dict = {}
    for mat in mgxs_lib.domain:
        mat_dict = {}
        for type_i in (t for t in mgxs_lib.mgxs_types if t not in ignored_types):                        
            data = mgxs_lib.get_mgxs(mat, type_i).get_xs() * conversion_factor
            if type_i == "nu-scatter matrix":
                data = data.transpose(2,1,0)
            mat_dict[type_i] = data

        mgxs_dict[mat.name] = mat_dict
    return mgxs_dict