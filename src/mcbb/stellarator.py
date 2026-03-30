from .blanket_base import OpenMCBlanketSimulation, Blanket, OpenMCSettingsBlanket
from .openmc_base_functions import _get_tally_results_divisor
from jax_sbgeom.flux_surfaces import ParametrisedSurface, ToroidalExtent
from jax_sbgeom.interfaces.blanket_creation import LayeredDiscreteBlanket
from dataclasses import dataclass
import copy
import os
from functools import cached_property
import jax.numpy as jnp
import numpy as onp
from typing import Literal, Union, List, Iterable, Dict
import jax_sbgeom as jsb
import openmc


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
class StellaratorOpenMCBlanketSimulation(OpenMCBlanketSimulation):
    parametrised_surface : ParametrisedSurface
    discrete_blanket     : LayeredDiscreteBlanket
    energy_bins          : Iterable[float]
    source_kwargs        : dict


    filename_start       : str
    boundary_type        : Literal['reflective', 'vacuum'] 

    @classmethod
    def from_blanket(cls, blanket : Blanket, discrete_blanket : LayeredDiscreteBlanket, parametrised_surface : ParametrisedSurface, energy_bins : Iterable[float], 
                     source_kwargs, 
                     batches : int, samples : int, download_location : os.PathLike, filename_start : str, boundary_type : Literal['reflective', 'vacuum'],
                     seed = None, verbosity = 7,
                     ):
        settings = OpenMCSettingsBlanket(download=True, download_location=download_location, seed = seed, batches = batches, samples = samples, verbosity = verbosity)        
        assert onp.allclose(onp.sort(energy_bins) , energy_bins), "Energy bins must be in ascending order"
        return cls(blanket, settings, parametrised_surface, discrete_blanket, energy_bins, source_kwargs, filename_start, boundary_type)

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
    def tallies(self):        
        tally_mesh_filename = self.filename_start + "_tally_source_mesh.h5m"

        tally_mesh_tetra = self.tally_mesh
        jsb.interfaces.dagmc_interface.tetrahedral_mesh_to_moab_mesh(*tally_mesh_tetra).write_file(tally_mesh_filename)

        umesh_tally = openmc.UnstructuredMesh(filename=tally_mesh_filename, library="moab", name = 'TallyMesh')
        mesh_filter = openmc.MeshFilter(umesh_tally)
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
        radial_source = jnp.nan_to_num(jsb.interfaces.plasma.flux_surface_reaction_rates_simple(self.discrete_blanket.s_spacing, **self.source_kwargs))

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
        return openmc.MeshSource(self.tallies[0].filters[0].mesh, openmc_mesh_sources)
        

    def get_tally_results(self, sp_file : Union[str, None] = None, quantities : List[Literal["mean", "std_dev", "rel_err"]] = ["mean"]):
        if sp_file is None:
            try:
                sp_file = self.sp_file
            except AttributeError:
                raise AttributeError("No statepoint file provided and simulation has not been run yet. Please run the simulation first or provide a specific statepoint file.")

        return _get_tally_results_divisor(sp_file, tuple(tally.name for tally in self.tallies), tuple(quantities))
    

    def _create_3d_importance_map(self, data_dict : Dict, n_samples : int):

        from .sn.base_sn import create_3D_importance_map_of_blanket
        self.blanket.thickn

        create_3D_importance_map_of_blanket(self.blanket, data_dict, )
    

    def create_fw_cadis_weight_window(self):
        pass
        

    # @cached_property    
    # def fast_flux(self):
    #     from blanket_parametrisation import GROUPS_LIST


    #     tally_results = self.get_tally_results()
    #     overlap_matrix = jax_sn.energy_set.compute_overlap_matrix(GROUPS_LIST, [15.2e6, 1e6])
        

    #     fast_flux = jnp.einsum("ij, kj -> ik", overlap_matrix, tally_results['FluxTally']['mean'] )[0]
    #     return fast_flux

def generate_flux_surface_source(flux_surface : ParametrisedSurface, s_spacing : jnp.ndarray, source_kwargs : Dict, n_theta : int, n_phi : int, toroidal_extent : ToroidalExtent, source_mesh_filename : str):
    '''
    Generates a flux_surface_source from a given radial coordinate spacing. Uses the jax_sbgeom flux_surfacde_reaction_rates_simple underneath.

    Parameters
    -----------
    flux_surface : ParametrisedSurface
        The flux surface to generate the source for. Used to generate the mesh and map the radial source onto the mesh.
    s_spacing : jnp.ndarray
        The radial coordinate spacing to generate the source for. Should be of shape (n_s,) where n_s is the number of radial points. The source will be generated at the midpoints of the radial spacing, so the first and last points will be ignored.
    source_kwargs : Dict
        The keyword arguments to pass to the jax_sbgeom flux_surface_reaction_rates_simple function. Should include at least the following keys:
            - "n0" : Central density
            - "nalpha" : Density shaping parameter
            - "Ti0" : Central ion temperature
            - "Tialpha" : Ion temperature shaping parameter

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

    radial_source = jnp.nan_to_num(jsb.interfaces.plasma.flux_surface_reaction_rates_simple(s_spacing, **source_kwargs))

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