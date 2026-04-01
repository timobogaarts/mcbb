import openmc 
import numpy as np
from typing import List, Iterable, Tuple, Literal
import jax.numpy as jnp

def create_openmc_settings(batches : int, samples : int, source : openmc.SourceBase, verbosity : int = 1, seed : int = None, weight_windows : openmc.WeightWindows = None):
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
    settings               =  openmc.Settings()
    settings.batches       = batches
    settings.particles     = int(samples)
    settings.inactive      = 0  #fixed source, so set 0 
    settings.verbosity     = verbosity

    if seed is not None:
        settings.seed          = seed
    else:
        settings.seed          = int(np.random.random(1)[0] * 1e13)    
    
    settings.run_mode      = 'fixed source'
    settings.source           = source
    settings.survival_biasing = False
    settings.output           = {'tallies' : False}    
    
    if weight_windows is not None:
        settings.weight_windows    = weight_windows
        settings.weight_windows_on =  True
    
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


def mgxs_lib_to_data_dicts(mgxs_lib : openmc.mgxs.Library, conversion_factor = 100.0):
    '''
    Function to convert an OpenMC mgxs.Library to a dictionary of multi-group cross-section data for easier use
    
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
    mgxs_dict = mgxs_lib_to_standard_formatting(mgxs_lib, conversion_factor)

    total_scat_data = {key : (mgxs_dict[key]['total'], mgxs_dict[key]['nu-scatter matrix']) for key in mgxs_dict.keys()}

    aux_data_dict = {key : {k : mgxs_dict[key][k] for k in mgxs_dict[key].keys() if k not in ['total', 'nu-scatter matrix']} for key in mgxs_dict.keys()}

    return total_scat_data, aux_data_dict
    


def _importance_map_to_weight_windows(importance_map : jnp.ndarray, ww_lower_upper_ratio : int = 3):
    '''
    Maps a importance map to weight windows.

    Non-meaningful values (e.g. nan values or negative values)are automatically set to the maximum weight window value. This is different than no weight windows.

    Parameters
    ----------
    importance_map : jnp.ndarray
        An importance map. Arbitrarily shaped (computation doesn't depend on specific shape)

    ww_lower_upper_ratio : int
        The ratio of the upper weight window to the lower weight window (default is 3)

    Returns
    -------
    ww_lower_norm : jnp.ndarray
        The normalised lower weight window values corresponding to the importance map
    ww_upper_norm : jnp.ndarray
        The normalised upper weight window values corresponding to the importance map
    '''
    

    importance_map_safe = jnp.where(importance_map > 0, importance_map, 1e-10)

    ww_lower     = jnp.where(importance_map > 0, 1/ ((ww_lower_upper_ratio + 1) / 2 * importance_map_safe) , 0)
    max_ww_value = jnp.max(ww_lower)

    ww_lower_norm  = ww_lower / (max_ww_value * (1.0 + ww_lower_upper_ratio ) /2.0 ) # normalise so avg is 1

    ww_lower_norm = jnp.where(ww_lower_norm == 0, jnp.max(ww_lower_norm), ww_lower_norm)
    
    
    return ww_lower_norm, ww_lower_norm * ww_lower_upper_ratio

def _convert_to_openmc_format(importance_map : jnp.ndarray, group_boundaries : jnp.ndarray):
    '''
    Flips the group boundaries to be in ascending order. Also puts the weight windows in the format expected by OpenMC ([..., n_groups]) instead of the format we use for computation ([n_groups, ...])

    Parameters
    ----------
    importance_map : jnp.ndarray
        Importance map. Assumed first axis is energy.
    group_boundaries : jnp.ndarray
        The energy group boundaries for the multi-group cross-section library, ordered from low to high

    Returns
    -------
    ww_lower_openmc : jnp.ndarray
        The lower weight window values corresponding to the importance map, in the format expected by OpenMC (group boundaries in descending order)
    
    ww_upper_openmc : jnp.ndarray
        The upper weight window values corresponding to the importance map, in the format expected by OpenMC (group boundaries in descending order)

    flipped_group_boundaries : jnp.ndarray
        The energy group boundaries in ascending order
    '''
    if np.all(np.sort(group_boundaries) == group_boundaries):
        ascending_group_boundaries = group_boundaries
        ascending_importance_map = importance_map
    else:
        ascending_group_boundaries = np.flip(group_boundaries, axis=0)
        ascending_importance_map = np.flip(importance_map, axis=0)

    flipped_transposed_importance_map = np.moveaxis(ascending_importance_map, 0, -1)

    return flipped_transposed_importance_map, ascending_group_boundaries


def importance_map_to_weight_window(importance_map : jnp.ndarray, group_boundaries : jnp.ndarray, mesh : openmc.MeshBase, ww_lower_upper_ratio : int = 3, **kwargs):
    '''
    Maps a importance map to weight windows for use in OpenMC.

    The importance map is transposed (energy groups last), and the group boundaries and the importance map are 

    Parameters
    ----------
    importance_map : jnp.ndarray
        An importance map. Should have the same shape as the mesh dimensions. [n_groups, mesh.shape]. Cannot be checked: openmc.MeshBase cannot a priori load a MOAB mesh, so 
        it doesn't know the mesh shape.

    group_boundaries : jnp.ndarray
        Energy groups used for the computation of the importance map

    mesh : openmc.MeshBase
        The OpenMC mesh object corresponding to the importance map.

    ww_lower_upper_ratio : int
        The ratio of the upper weight window to the lower weight window (default is 3)

    **kwargs: dict

        Directly passed to openmc.WeightWindows.

    Returns
    -------
    weight_windows : openmc.WeightWindows
        An OpenMC WeightWindows object containing the lower and upper weight windows corresponding to the importance map.
    '''
    assert importance_map.shape[0] == np.array(group_boundaries).shape[0] - 1, "The first dimension of the importance map should match the number of energy groups (group boundaries - 1)"

    openmc_importance_map, group_boundaries_openmc = _convert_to_openmc_format(importance_map, group_boundaries)
    ww_lower_norm, ww_upper_norm                   = _importance_map_to_weight_windows(openmc_importance_map, ww_lower_upper_ratio)    #[mesh.shape, n_groups]            
    weight_windows = openmc.WeightWindows(mesh, lower_ww_bounds = np.array(ww_lower_norm), upper_ww_bounds = np.array(ww_upper_norm), energy_bounds = np.array(group_boundaries_openmc), particle_type = "neutron", **kwargs)

    return weight_windows


def tetrahedral_mesh_to_dagmc_file(mesh : Tuple[jnp.ndarray, jnp.ndarray], filename : str, conversion_factor : float = 100.0):
    '''
    Function to convert a tetrahedral mesh (given by vertices and connectivity) to a DAGMC .h5m file for use in OpenMC.

    Parameters
    ----------
    mesh : Tuple[jnp.ndarray, jnp.ndarray]
        A tuple containing the vertices and connectivity of the tetrahedral mesh. The vertices should be of shape [n_vertices, 3] and the connectivity should be of shape [n_elements, 4] (assuming tetrahedral elements).
    filename : str
        The name of the output .h5m file to create

    Returns
    -------
    filename
        This function creates a .h5m file type at the specified filename location and returns the filename
    '''
    import h5py
    assert len(mesh) == 2, "Mesh should be a tuple of (vertices, connectivity)"
    assert mesh[0].shape[1] == 3, "Vertices should be of shape [n_vertices, 3]"
    assert mesh[1].shape[1] == 4, "Connectivity should be of shape [n_elements, 4] for tetrahedral elements"
    from jax_sbgeom.interfaces.dagmc_interface import tetrahedral_mesh_to_moab_mesh

    tetrahedral_mesh_to_moab_mesh(*mesh, conversion_factor).write_file(filename)    
    return filename


def tetrahedral_mesh_to_openmc_mesh(mesh : Tuple[jnp.ndarray, jnp.ndarray], filename : str, name : str, conversion_factor : float = 100.0):
    '''
    Function to convert a tetrahedral mesh (given by vertices and connectivity) to an OpenMC UnstructuredMesh object, using a DAGMC .h5m file.
    
    Also applies a conversion factor to the vertex coordinates (default is 100.0 to convert from m to cm, which is the unit expected by OpenMC)

    Parameters
    ----------
    mesh : Tuple[jnp.ndarray, jnp.ndarray]
        A tuple containing the vertices and connectivity of the tetrahedral mesh. The vertices should be of shape [n_vertices, 3] and the connectivity should be of shape [n_elements, 4] (assuming tetrahedral elements).
    filename : str  
        The name of the output .h5m file to create for the DAGMC mesh
    name : str
        The name to give to the OpenMC UnstructuredMesh object
    conversion_factor : float
        A conversion factor to apply to the vertex coordinates (default is 100.0 to convert from m to cm, which is the unit expected by OpenMC)
    
    Returns
    -------
    openmc.UnstructuredMesh
        An OpenMC UnstructuredMesh object created from the tetrahedral mesh, with the specified name and vertex coordinates converted using the conversion factor.
    '''    
    tetrahedral_mesh_to_dagmc_file(mesh, filename, conversion_factor)
    return openmc.UnstructuredMesh( filename = filename, library = 'moab', name = name)

def tetrahedral_mesh_importance_to_openmc_weight_window(mesh : Tuple[jnp.ndarray, jnp.ndarray], importance_map : jnp.ndarray, group_boundaries : jnp.ndarray, filename : str, name : str, conversion_factor : float =100.0, ww_lower_upper_ratio : int = 3, **kwargs):
    assert importance_map.shape[-1] == mesh[1].shape[0], f"The last dimension of the importance map: {importance_map.shape[-1]} should match the number of elements in the mesh {mesh[1].shape[0]}"

    openmc_mesh = tetrahedral_mesh_to_openmc_mesh(mesh, filename, name, conversion_factor)
    weight_windows = importance_map_to_weight_window(importance_map, group_boundaries = group_boundaries, mesh = openmc_mesh, ww_lower_upper_ratio = ww_lower_upper_ratio, **kwargs)
    return weight_windows

import functools
import os
@functools.lru_cache(maxsize=2)
def _get_tally_results_divisor(sp_file : os.PathLike,  tally_names : Iterable[str], quantities : List[Literal["mean", "std_dev", "rel_err"]] = ["mean"]):
    result = {}
    
    with openmc.StatePoint(sp_file) as sp:           
        for tally in tally_names:
            sp_tally = sp.get_tally(name = tally) 
            mesh_data    = sp_tally.find_filter(openmc.MeshFilter).mesh # assume all are on mesh
            result[sp_tally.name] = {}
            for quantity in quantities:
                if quantity not in ["mean", "std_dev", "rel_err"]:
                    raise ValueError(f"Quantity {quantity} not recognized. Valid options are 'mean', 'std_dev', and 'rel_err'.")
                result_key = f"{tally}_{quantity}"

                if quantity == "rel_err":
                    divisor = 1.0
                else:
                    divisor = np.abs(mesh_data.volumes[:, None]) #[vol, eg]

                
                result[sp_tally.name][quantity] = np.flip(sp_tally.get_reshaped_data(quantity)[..., 0,0],axis=1) / divisor # flip to descending again

    return result