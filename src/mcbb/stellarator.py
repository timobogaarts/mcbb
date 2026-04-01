from mcbb.linear_blanket import Blanket1D

from .blanket_base import OpenMCBlanketSimulation, Blanket, OpenMCSettingsBlanket, _core_dict
from .openmc_base_functions import _get_tally_results_divisor
from jax_sbgeom.flux_surfaces import ParametrisedSurface, ToroidalExtent, FluxSurface, FluxSurfaceExtendedDistanceMatrix, FluxSurfaceNormalExtendedNoPhi
from jax_sbgeom.interfaces.blanket_creation import LayeredDiscreteBlanket
from dataclasses import dataclass
import copy
import os
from functools import cached_property
import jax.numpy as jnp
import numpy as onp
from typing import Literal, Union, List, Iterable, Dict, Callable
import jax_sbgeom as jsb
import openmc
from .blanket_base import BlanketLayer, Blanket


def create_openmc_sector_region_dagmc(filename, toroidal_extent : ToroidalExtent, boundary_type : Literal['reflective', 'vacuum']):
    import openmc

    phi1 = toroidal_extent.start 
    phi2 = toroidal_extent.end

    plane1 = openmc.Plane( - onp.sin(phi1), onp.cos(phi1), 0.0, 0.0, boundary_type=boundary_type)
    plane2 = openmc.Plane( - onp.sin(phi2), onp.cos(phi2), 0.0, 0.0, boundary_type=boundary_type)

    dagmc_univ = openmc.DAGMCUniverse(filename  =  filename, auto_geom_ids=True)
    bounding_box = dagmc_univ.bounding_region()    
    region = +plane1 & -plane2 &  bounding_box
        
    return openmc.Geometry(root = [openmc.Cell(cell_id = 9999, region = region, fill = dagmc_univ)])

@dataclass
class StellaratorBlanketLayer(BlanketLayer):
    physical_thickness_matrix : jnp.ndarray
    '''
    The physical distance matrix of a stellarator blanket layer. [0,0] corresponds to theta, phi = 0,0, [-1,-1] corresponds to (2pi , 2pi / nfp)
    '''

@dataclass
class StellaratorBlanket(Blanket):
    
    layers : List[StellaratorBlanketLayer]    
    
    @classmethod 
    def from_1D_blanket(cls, blanket_1D : Blanket1D,  physical_d_matrices : List[jnp.ndarray]):
        assert len(blanket_1D.layers) == len(physical_d_matrices), "Length of physical_d_matrices must match number of layers in blanket_1D"
        layers = []
        for layer_1D, thickness_matrix in zip(blanket_1D.layers, physical_d_matrices):
            layers.append(StellaratorBlanketLayer(name = layer_1D.name, elements = layer_1D.elements, physical_thickness_matrix = jnp.atleast_2d(thickness_matrix)))
        return cls(layers)

    @cached_property     
    def average_thicknesses(self):
        return jnp.array([jnp.mean(layer.physical_thickness_matrix) for layer in self.layers[:]])

    @property
    def thicknesses(self):
        return [layer.physical_thickness_matrix for layer in self.layers]
    

@dataclass
class StellaratorOpenMCBlanketSimulation(OpenMCBlanketSimulation):
    blanket              : StellaratorBlanket
    parametrised_surface : ParametrisedSurface
    discrete_blanket     : LayeredDiscreteBlanket
    energy_bins          : Iterable[float]
    source_callable      : Callable[[jnp.ndarray], jnp.ndarray]  # A callable that takes in a radial coordinate spacing and returns a radial source distribution

    filename_start       : str
    boundary_type        : Literal['reflective', 'vacuum']

    @classmethod
    def from_blanket(cls, blanket : Blanket, discrete_blanket : LayeredDiscreteBlanket, parametrised_surface : ParametrisedSurface, energy_bins : Iterable[float], 
                     source_callable : Callable[[jnp.ndarray], jnp.ndarray],
                     batches : int, samples : int, download_location : os.PathLike, filename_start : str, boundary_type : Literal['reflective', 'vacuum'],
                     seed = None, verbosity = 7,
                     ):
        settings = OpenMCSettingsBlanket(download=True, download_location=download_location, seed = seed, batches = batches, samples = samples, verbosity = verbosity)        
        assert onp.allclose(onp.sort(energy_bins) , energy_bins), "Energy bins must be in ascending order"
        return cls(blanket, settings, parametrised_surface, discrete_blanket, energy_bins, source_callable, filename_start, boundary_type)

    @cached_property
    def geometry(self):
        names = [layer.name for layer in self.blanket.layers][1:]        
        dagmc_surface_model = jsb.interfaces.dagmc_interface.create_dagmc_surface_mesh(self.discrete_blanket, self.parametrised_surface, names)
        dagmc_surface_model_filename =  self.filename_start + "_dagmc_geometry.h5m"
        dagmc_surface_model.write_file(dagmc_surface_model_filename)
        return create_openmc_sector_region_dagmc(dagmc_surface_model_filename,self.discrete_blanket.toroidal_extent, self.boundary_type)
    
    @cached_property
    def tally_mesh(self):
        return self.discrete_blanket.volume_mesh(self.parametrised_surface)
    
    @cached_property 
    def tally_mesh_openmc(self):
    
        tally_mesh_filename = self.filename_start + "_tally_source_mesh.h5m"

        tally_mesh_tetra = self.tally_mesh
        jsb.interfaces.dagmc_interface.tetrahedral_mesh_to_moab_mesh(*tally_mesh_tetra).write_file(tally_mesh_filename)
        print(tally_mesh_filename)
        return openmc.UnstructuredMesh(filename=tally_mesh_filename, library="moab", name = 'BaseMesh')

        
    def with_weight_windows(self, weight_windows : openmc.WeightWindows):        
        raise NotImplementedError("This method is not implemented for the StellaratorOpenMCBlanketSimulation class. Use add_weight_windows instead to ensure the weight window mesh is the same as the tally mesh.")
    
    def add_weight_windows(self, weight_windows : openmc.WeightWindows):
        '''
        Add weight windows to the simulation. This method is used instead of with_weight_windows to ensure that the weight windows are added to the same mesh as the tally and source.

        Parameters
        ----------
        weight_windows : openmc.WeightWindows
            The weight windows to use in the new simulation object
        Returns
        -------
        new_simulation : OpenMCBlanketSimulation
            A new OpenMCBlanketSimulation object with the same geometry, materials, and tallies as the original, but with the specified weight windows in the settings
        -------
        
        '''
        try:
            del self.settings
        except AttributeError:
            pass # settings have not been created yet, so we can just set the weight windows without worrying about deleting the cached property
        self.settings_blanket.weight_windows = weight_windows
    
    @cached_property
    def tallies(self):        
        
        mesh_filter = openmc.MeshFilter(self.tally_mesh_openmc)
        energy_filter = openmc.EnergyFilter(self.energy_bins)

        tally_flux = openmc.Tally(name='FluxTally')
        tally_flux.filters = [mesh_filter, energy_filter]
        tally_flux.scores = ['flux']
        tally_flux.estimator = "tracklength"

        tally_tbr = openmc.Tally(name="TBRTally")
        tally_tbr.filters = [mesh_filter, energy_filter]
        tally_tbr.scores = ['(n,Xt)']
        tally_tbr.estimator = "tracklength"

        return openmc.Tallies([tally_flux, tally_tbr])

    @cached_property
    def source(self):    
        '''
        Radial source is generated using the jax_sbgeom flux_surface_reaction_rates_simple function, which takes the radial coordinate spacing and the source_kwargs as input.
        Source generated is the same as the tally mesh.

        '''
        radial_source = jnp.nan_to_num(self.source_callable(self.discrete_blanket.s_spacing))

        radial_source_midpoint = (radial_source[:-1] + radial_source[1:])/2
        source_on_mesh            = self.discrete_blanket.volume_mesh_structure.map_radial_array_to_layers(radial_source_midpoint)

        tet_volumes = jsb.jax_utils.mesh.volumes_tetrahedra(*self.tally_mesh)


        centre_tetrahedral_plasma_mesh = [float(i * 100.0) for i in onp.mean(self.tally_mesh[0], axis=0)]

        # Build template once — space/angle/energy are identical for all elements
        template = openmc.IndependentSource()
        # space is ignored when sampling from MeshSource, but still checked for geometry validity
        template.space = openmc.stats.Point(centre_tetrahedral_plasma_mesh)
        template.angle = openmc.stats.Isotropic()
        template.energy = openmc.stats.Discrete([14.1e6], [1.0])  # 14.1 MeV neutrons

        def make_source(rate):
            src = copy.copy(template)
            src.strength = float(rate)
            return src
        reaction_rates_volume_weighted = onp.array(source_on_mesh * tet_volumes)
        
        openmc_mesh_sources            = onp.frompyfunc(make_source, 1, 1)(reaction_rates_volume_weighted)        
        return openmc.MeshSource(self.tally_mesh_openmc, openmc_mesh_sources)
    def _norm_sp_file(self,  sp_file : Union[os.PathLike, None] ):
        if sp_file is None:
            try:
                sp_file = self.sp_file
            except AttributeError:
                raise AttributeError("No statepoint file provided and simulation has not been run yet. Please run the simulation first or provide a specific statepoint file.")
        return sp_file

    def get_tally_results(self, sp_file : Union[os.PathLike, None] = None, quantities : List[Literal["mean", "std_dev", "rel_err"]] = ["mean"]):
        
        sp_file = self._norm_sp_file(sp_file)
        
        return _get_tally_results_divisor(sp_file, tuple(tally.name for tally in self.tallies), tuple(quantities))
    
    def _create_1d_importance_map(self, data_dict : Dict, distance_samples : jnp.ndarray, degree = 3, tn_order = 3, n_elem_per_region = 5, **kwargs):
        from .sn.base_sn import create_importance_map_for_blanket            
        return create_importance_map_for_blanket(self.blanket, data_dict, [n_elem_per_region for i in range(len(self.blanket.layers))], degree = degree, tn_order = tn_order,s_values_blanket = distance_samples, **kwargs)

    def _create_1d_importance_map_radial(self, data_dictionary : Dict, degree = 3, tn_order = 3, n_elem_per_region = 5, **kwargs):

        d_physical = self.discrete_blanket.map_to_physical_spacing(jnp.cumsum(jnp.array(self.blanket.thicknesses)))

        return d_physical, self._create_1d_importance_map(data_dictionary, d_physical, degree, tn_order, n_elem_per_region, **kwargs)
    
    def create_weight_windows(self, data_dictionary : Dict, energy_groups : Iterable[float], degree  : int = 3, tn_order : int = 3, n_elem_per_region : int = 5, ww_lower_upper_ratio : int = 3, ww_kwargs : Dict = {"max_split" : 2000}, **kwargs):
        '''
        Creates weight windows for the blanket based on a forward-weighted CADIS importance map of a 1D blanket using the average thicknesses. 
        The weight windows are on the same mesh as the tally and the source. This reduces overhead of the mesh localization.

        Parameters:
        ----------
        data_dictionary : Dict
            A dictionary containing any additional data required to create the importance map. This is assumed to have the same keys as the blanket.layer_names property. The dictionary 
            values are a tuple of (total_cross_section, scattering_cross_section), where total_cross section is of shape [n_groups] and scattering_cross_section is of shape [l_order, n_groups, n_groups].
        
            

        degree : int
            The degree of the finite element basis to use for the importance map generation. Default is 3.
        tn_order : int
            The order of the Tn quadrature to use for the importance map generation. Default is 3.
        n_elem_per_region : int
            The number of finite elements to use per region in the importance map generation. Default is 5.
        ww_lower_upper_ratio : int
            The ratio of the lower to upper weight window values. Default is 3, meaning the upper weight window value is 3 times the lower weight window value. 
        **kwargs : dict
            Any additional keyword arguments to pass to the importance map generation function (which passes it directly to the multigroup operator creation, so it can be used to set different operators for the 
            multigroup operator, e.g. TransportOperatorVmapPallas instead of TransportOperatorVmap).
        '''
        from .openmc_base_functions import importance_map_to_weight_window
        assert onp.all(onp.sort(energy_groups)[::-1] == onp.array(energy_groups)), "Energy groups must be in descending order for this function (it uses the same ordering as jax-sn, namely descending). It is then automatically converted to OpenMC format (ascending)"
        d_physical, importance_map = self._create_1d_importance_map_radial(data_dictionary, degree, tn_order, n_elem_per_region, **kwargs)

        importance_map = jnp.where(d_physical < self.blanket.average_thicknesses[0], jnp.nan, importance_map) # set importance map to nan in the plasma region, as we do not want to generate weight windows there.
        # They will be automatically set to the maximum weight window value.

        importance_map_element_wise = 0.5 * (importance_map[..., 1:] + importance_map[..., :-1]) # map to element-wise values by averaging the importance map at the element boundaries. 

        return importance_map_to_weight_window(self.discrete_blanket.volume_mesh_structure.map_radial_array_to_layers(importance_map_element_wise), energy_groups, self.tally_mesh_openmc, ww_lower_upper_ratio, **ww_kwargs)


    
    def fast_flux(self, energy_groups, sp_file : Union[os.PathLike, None] = None):
        from jax_sn.energy_set import compute_overlap_matrix
        sp_file = self._norm_sp_file(sp_file)
        
        tally_results = self.get_tally_results(sp_file)
        overlap_matrix = compute_overlap_matrix(energy_groups, [15.2e6, 1e6])
        

        fast_flux = jnp.einsum("ij, kj -> ik", overlap_matrix, tally_results['FluxTally']['mean'] )[0]
        return fast_flux

def generate_flux_surface_source(flux_surface : ParametrisedSurface, s_spacing : jnp.ndarray, source_callable : Callable[[jnp.ndarray], jnp.ndarray], n_theta : int, n_phi : int, toroidal_extent : ToroidalExtent, source_mesh_filename : str):
    '''
    Generates a flux_surface_source from a given radial coordinate spacing.

    An example to generate the callable source is to use the jax_sbgeom function jax_sbgeom flux_surface_reaction_rates_simple function.
    
    `source_callable = lambda s: jsb.flux_surfaces.flux_surface_reaction_rates_simple(s, n0 = 1e20, nalpha = 2.0, Ti0 = 10e3, Tialpha = 2.0)`
    - "n0" : Central density
    - "nalpha" : Density shaping parameter
    - "Ti0" : Central ion temperature
    - "Tialpha" : Ion temperature shaping parameter

    Automatically applies a conversion factor of 100.0 to convert from meters to centimeters, as OpenMC and MOAB use centimeters as the default unit for geometry.


    Parameters
    -----------
    flux_surface : ParametrisedSurface
        The flux surface to generate the source for. Used to generate the mesh and map the radial source onto the mesh.
    s_spacing : jnp.ndarray
        The radial coordinate spacing to generate the source for. Should be of shape (n_s,) where n_s is the number of radial points. The source will be generated at the midpoints of the radial spacing, so the first and last points will be ignored.
    source_callable : Callable[[jnp.ndarray], jnp.ndarray]
        A callable that takes in a radial coordinate spacing and returns a radial source distribution.
     
    n_theta : int
        The number of poloidal points to use for the mesh.
    n_phi : int
        The number of toroidal points to use for the mesh.
    toroidal_extent : ToroidalExtent
        The toroidal extent of the mesh. Should be a ToroidalExtent object.
    source_mesh_filename : str
        The filename to save the generated mesh to. Should be a string ending in .h5m, as the mesh will be saved in MOAB format.    
    
    Returns
    --------
    openmc.MeshSource
        An OpenMC MeshSource object containing the generated source. 
    
        
    '''

    radial_source = jnp.nan_to_num(source_callable(s_spacing))

    radial_source_midpoint = (radial_source[:-1] + radial_source[1:])/2

    source_mesh            = jsb.flux_surfaces.mesh_tetrahedra(flux_surface, s_spacing, n_theta, n_phi, toroidal_extent)


    structure              = jsb.interfaces.blanket_creation.BlanketMeshStructure(n_theta, n_phi, s_spacing.shape[0], True, toroidal_extent.full_angle())

    source_on_mesh         = structure.map_radial_array_to_layers(radial_source_midpoint)

    
    tet_volumes = jsb.jax_utils.mesh.volumes_tetrahedra(*source_mesh)


    centre_tetrahedral_plasma_mesh = [float(i * 100.0) for i in onp.mean(source_mesh[0], axis=0)]

    jsb.interfaces.dagmc_interface.tetrahedral_mesh_to_moab_mesh(*source_mesh).write_file(source_mesh_filename)

    umesh_source = openmc.UnstructuredMesh(filename=source_mesh_filename, library="moab", name = 'SourceMesh')



    # Build template once — space/angle/energy are identical for all elements
    template = openmc.IndependentSource()
    # space is ignored when sampling from MeshSource, but still checked for geometry validity
    template.space = openmc.stats.Point(centre_tetrahedral_plasma_mesh)
    template.angle = openmc.stats.Isotropic()
    template.energy = openmc.stats.Discrete([14.1e6], [1.0])  # 14.1 MeV neutrons

    def make_source(rate):
        src = copy.copy(template)
        src.strength = float(rate)
        return src
    reaction_rates_volume_weighted = onp.array(source_on_mesh * tet_volumes)
    
    openmc_mesh_sources            = onp.frompyfunc(make_source, 1, 1)(reaction_rates_volume_weighted)        
    return openmc.MeshSource(umesh_source, openmc_mesh_sources)