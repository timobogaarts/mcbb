import jax.numpy as jnp
from ..blanket_base import Blanket, OpenMCBlanketSimulation
from typing import Dict, Tuple 
import lineax as lx

def setup_linear_symm_mgop(blanket : Blanket, data_dictionary : Dict[str, Tuple[jnp.ndarray, jnp.ndarray]], 
               n_elem_per_region : Tuple[int],
               degree : int,
               tn_order : int,
               **kwargs):    
    import jax_sn    
    from jax_sn.domain.geometric_domain import CrossSectionData
    from jax_sn.domain.domain_generation import create_cartesian_setup
    from jax_sn.domain import Domain
    from jax_sn.solution_domain import SolutionDomain, BasixLagrangianSimplex    
    from jax_sn.quadrature_set import create_tn_quadrature_set

    xs_data_dict = {name : CrossSectionData(total = jnp.array(data_dictionary[name][0]), scattering=jnp.array(data_dictionary[name][1])) for name in data_dictionary.keys()}

    names_blanket = [layer.name for layer in blanket.layers]

    layer_sizes = [layer.thickness for layer in blanket.layers]
    
    assert set(names_blanket) == set(xs_data_dict.keys()), "The names of the layers in the blanket must match the names of the cross-section data. Names {} and data names {}".format(names_blanket, xs_data_dict.keys())

    xs_regions = [xs_data_dict[name] for name in names_blanket]
    region_size_res = [(size, res) for size, res in zip(layer_sizes, n_elem_per_region)]

    def _symmetrize_no_plasma(list):
        return [list[-( i + 1)] for  i in range(len(list) - 1)] + list
    
    region_size_res_symm = _symmetrize_no_plasma(region_size_res)
    source_eg0_symm      = _symmetrize_no_plasma([1 if i ==0 else 0 for i in range(len(names_blanket))])    
    xs_regions_symm      = _symmetrize_no_plasma(xs_regions)


    verts, conn, cross_section_idx, source = create_cartesian_setup(1, [region_size_res_symm], xs_regions_symm, source_eg0_symm,0)

    element = BasixLagrangianSimplex(degree=degree, dimension = 1)

    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(
        element = element,
        domain = Domain.from_mesh_and_cross_sections((verts, conn), element.face_template, cross_section_idx)
    )

    source_basis = jnp.repeat(source, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)

    return jax_sn.operators.multi_group_operator.create_multi_group_operator(solution_domain, create_tn_quadrature_set(tn_order), **kwargs), source_basis

    
    
def create_weight_windows_for_blanket(blanket : Blanket, data_dictionary : Dict[str, Tuple[jnp.ndarray, jnp.ndarray]], 
               n_elem_per_region : Tuple[int],
               degree : int,
               tn_order : int,               
               n_weight_windows_in_blanket : int ,
               solver : lx.AbstractLinearSolver = lx.BiCGStab(1e-10, 1e-10, max_steps = 200),               
               **kwargs):
    import jax_sn
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
    n_weight_windows_in_blanket : int,
        Number of weight window sampling nodes in the blanket
    solver : lx.AbstractLinearSolver, optional
        The linear solver to use for solving the forward-weighted CADIS equations. Default is BiCGStab with a tolerance of 1e-10 and a maximum of 200 steps.
    '''
    
    mgop, source_basis = setup_linear_symm_mgop(blanket, data_dictionary, n_elem_per_region, degree, tn_order, **kwargs)
    
    element = jax_sn.solution_domain.BasixLagrangianSimplex(degree, 1)
    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(element, mgop.domain)
    rhs = jax_sn.operators.multi_group_operator.right_hand_side.IsotropicSourceSingleGroup.from_operator(mgop, source_basis)

    importance_map_solution = jax_sn.variance_reduction.fw_cadis_scalar_flux(mgop, rhs, solver)

    blanket_start = sum([layer.thickness for layer in blanket.layers])
    blanket_end = sum([layer.thickness for layer in blanket.layers[1:]])+ blanket_start
    
    d_interp = jnp.linspace(blanket_start, blanket_end, n_weight_windows_in_blanket)
    interpolated_importance_map = solution_domain.interpolate(d_interp[:, None], importance_map_solution[:, 0 , ...])    
    
    return d_interp - blanket_start,  interpolated_importance_map
    
    
    

    