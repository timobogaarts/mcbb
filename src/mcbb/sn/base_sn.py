import jax.numpy as jnp
from ..blanket_base import Blanket, OpenMCBlanketSimulation
from typing import Dict, Tuple 
import lineax as lx

def setup_linear_symm_mgop(blanket : Blanket, data_dictionary : Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
               n_elem_per_region : Tuple[int],
               degree : int,
               tn_order : int,
               multi_group_strategy = None,
               **sweep_data_kwargs):
    '''
    **sweep_data_kwargs
        Forwarded as ``sweep_data_kwargs`` to ``create_multi_group_operator``.
    '''
    import jax_sn
    from jax_sn.domain.geometric_domain import CrossSectionData
    from jax_sn.domain.domain_generation import create_cartesian_setup
    from jax_sn.domain import Domain
    from jax_sn.solution_domain import SolutionDomain, BasixLagrangianSimplex
    from jax_sn.quadrature_set import create_tn_quadrature_set
    from jax_sn.operator_creation.multi_group_operator_creation import create_multi_group_operator
    from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy

    if multi_group_strategy is None:
        multi_group_strategy = DefaultMultiGroupStrategy.default_multi_group_strategy(checkpoint=False)

    xs_data_dict = {name : CrossSectionData(total = jnp.array(data_dictionary[name][0]), scattering=jnp.array(data_dictionary[name][1])) for name in data_dictionary.keys()}

    names_blanket = [layer.name for layer in blanket.layers]

    layer_sizes =  blanket.average_thicknesses
    
    assert set(names_blanket) == set(xs_data_dict.keys()), "The names of the layers in the blanket must match the names of the cross-section data. Names {} and data names {}".format(names_blanket, xs_data_dict.keys())

    xs_regions = [xs_data_dict[name] for name in names_blanket]
    region_size_res = [(size, res) for size, res in zip(layer_sizes, n_elem_per_region)]

    def _symmetrize_no_plasma(seq):
        return [seq[-( i + 1)] for  i in range(len(seq) - 1)] + seq
    
    region_size_res_symm = _symmetrize_no_plasma(region_size_res)
    source_eg0_symm      = _symmetrize_no_plasma([1 if i ==0 else 0 for i in range(len(names_blanket))])    
    xs_regions_symm      = _symmetrize_no_plasma(xs_regions)


    verts, conn, cross_sections_indexed, source = create_cartesian_setup(1, [region_size_res_symm], xs_regions_symm, source_eg0_symm,0)

    element = BasixLagrangianSimplex(degree=degree, dimension = 1)

    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(
        element = element,
        domain = Domain.from_mesh_and_cross_sections((verts, conn), element.face_template, cross_sections_indexed)
    )

    source_basis = jnp.repeat(source, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)

    reduced_quadrature_set = jax_sn.quadrature_set.QuadratureSetReduced.from_quadrature_set(create_tn_quadrature_set(tn_order), solution_domain.dimension)

    mgop = create_multi_group_operator(solution_domain, reduced_quadrature_set, multi_group_strategy, sweep_data_kwargs=sweep_data_kwargs)

    return mgop, source_basis, solution_domain

    
    
def create_importance_map_for_blanket(blanket : Blanket, data_dictionary : Dict[str, Tuple[jnp.ndarray, jnp.ndarray]], 
               n_elem_per_region : Tuple[int],
               degree : int,
               tn_order : int,               
               s_values_blanket : jnp.ndarray,
               solver : lx.AbstractLinearSolver = lx.BiCGStab(1e-10, 1e-10, max_steps = 200),               
               **kwargs):
    '''
    Automated 1D weight window generation using a 1D linear symmetric domain.

    Parameters:
    ------------
    blanket : Blanket
        The blanket for which to generate weight windows.
    data_dictionary : Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
        A dictionary mapping layer names to their cross-section data. Each entry should be a tuple of (total_cross_section, scattering_cross_section), where total_cross section is of shape [n_groups] and scattering_cross_section is of shape [l_order, n_groups, n_groups].
    n_elem_per_region : Tuple[int]
        A tuple specifying the number of finite elements to use in each region of the blanket. The length of this tuple should match the number of layers in the blanket.
    degree : int
        The degree of the finite element basis functions to use.
    tn_order : int
        The order of the Tn quadrature set to use for angular discretization.
    s_values_blanket : jnp.ndarray
        The spatial locations within the blanket at which to evaluate the importance map, relative to the start of the blanket (shape [n_weight_windows_in_blanket])
        Includes the plasma region: the blanket.average_thicknesses[0] determines the extent to which the plasma is taken. If the plasma is not desired,
        one should set the first entry of s_values_blanket to be equal to blanket.average_thicknesses[0] and increase from there. 

    solver : lx.AbstractLinearSolver, optional
        The linear solver to use for solving the forward-weighted CADIS equations. Default is BiCGStab with a tolerance of 1e-10 and a maximum of 200 steps.
    **kwargs
        Additional keyword arguments to pass to the multi-group operator setup function.(setup_linear_symm_mgop)

    Returns
    -------
    d_interp : jnp.ndarray
        The spatial locations of the weight windows within the blanket, relative to the start of the blanket (shape [n_weight_windows_in_blanket])
    interpolated_importance_map : jnp.ndarray
        The interpolated importance map values at the weight window locations (shape [n_groups, n_weight_windows_in_blanket])
    '''
    
    from jax_sn.solvers.right_hand_side import IsotropicSourceSingleGroup
    from jax_sn.variance_reduction import fw_cadis_scalar_flux

    mgop, source_basis, solution_domain = setup_linear_symm_mgop(blanket, data_dictionary, n_elem_per_region, degree, tn_order, **kwargs)

    rhs = IsotropicSourceSingleGroup.from_operator(mgop, source_basis)

    importance_map_solution = fw_cadis_scalar_flux(mgop, rhs, solver)

    blanket_start = sum(blanket.average_thicknesses[1:])   # compensate for symmetry!
    
    interpolated_importance_map = solution_domain.interpolate(blanket_start +s_values_blanket[:, None], importance_map_solution[:, 0 , ...])    
    
    return interpolated_importance_map

def create_3D_importance_map_of_blanket(
                               blanket_base : Blanket,
                               data_dict : dict,                                                       
                               n_s : int, n_theta : int, n_phi : int, toroidal_extent, parametrised_surface ,                                      
                               degree : int = 3,
                               tn_order : int = 3,                               
                               disc_per_region : int = 5,                               
                               ):
    from jax_sbgeom.flux_surfaces import ParametrisedSurface, ToroidalExtent, mesh_tetrahedra

    assert isinstance(blanket_base, Blanket), "blanket_base must be an instance of Blanket"
    assert isinstance(parametrised_surface, ParametrisedSurface), "parametrised_surface must be an instance of jax_sbgeom.ParametrisedSurface"
    assert isinstance(toroidal_extent, ToroidalExtent), "toroidal_extent must be an instance of jax_sbgeom.ToroidalExtent"

    s_values_blanket, importance_map = create_importance_map_for_blanket(blanket_base, data_dict, [disc_per_region for i in range(len(blanket_base.layers))], degree, tn_order, n_weight_windows_in_blanket= n_s) #[n_s], [n_energy_groups]
    fw_distance = jnp.maximum(0.0, blanket_base.average_thicknesses[0] - 1.0) # fw distance is encoded as the distanc beyond 1 in a 

    tet_mesh    = mesh_tetrahedra(parametrised_surface, 1.0 + fw_distance + s_values_blanket, toroidal_extent, n_theta, n_phi)    

    importance_map_layer = 0.5 *(importance_map[..., :-1] + importance_map[..., 1:]) # [n_layer_blocks]

    if toroidal_extent.full_angle():
        element_shape = (n_s -1, n_theta, n_phi, 6)
    else:
        element_shape = (n_s -1, n_theta, n_phi - 1, 6)

    assert jnp.prod(jnp.array(element_shape)) == tet_mesh[1].shape[0], f"Element shape {element_shape} does not match the number of elements in the mesh {tet_mesh[1].shape[0]}"
    
    importance_map_3d = jnp.broadcast_to(importance_map_layer[:,:, None, None, None], (importance_map.shape[0],)+element_shape)
    
    return tet_mesh, importance_map_3d.reshape(importance_map.shape[0], -1)

