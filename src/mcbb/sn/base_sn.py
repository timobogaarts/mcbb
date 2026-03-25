import jax.numpy as jnp
from ..blanket_base import Blanket, OpenMCBlanketSimulation
from typing import Dict, Tuple 


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

    
    

    
    

    