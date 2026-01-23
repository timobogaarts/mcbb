import os 
import contextlib
from typing import List
from .blanket_base import Blanket
def create_openmc_ce_materials(blanket : Blanket, download=False, download_location = "",  libraries  : List[str] = ["FENDL-3.1d"], print_output = False) -> dict:
    import openmc
    import openmc_data_downloader
    '''
    Function to create continuous-energy materials for OpenMC from a dictionary of materials

    It also sets the OPENMC_CROSS_SECTIONS variable.

    Parameters
    ----------
    material_dictionary : dict
        The material dictionary
    download : bool
        If True, download the cross-section data for the materials
        If False, use the cross-section data already downloaded but still set the OPENMC_CROSS_SECTIONS environment variable
    download_location : str
        The location where the cross-section data should be downloaded to
    libraries : List[str]
        A list of libraries to download the cross-section data from. Default is ["FENDL-3.1d"]
    
    Returns
    -------    
    total_material_dict : dict
        A dictionary containing the OpenMC materials    
    -------
    
    The material dictionary should be in the following format:    
    {
        "Material_name": {
            "Elements": {
                "Element_name": atom_number,
                ...
            }
        },
        ...
    }

    The atom_number is the number of atoms per m^3
    It can also contain other fields used for the specific benchmark (e.g. a thickness of the material in a slab geometry), but this will be ignored here

    '''

    total_material_dict = []
    for i, layer in enumerate(blanket.layers):
        mat_i = openmc.Material(name = layer.name)
        elements_i = layer.elements
        total_atom_number = sum(elements_i.values())
        for atom in elements_i.keys():
            nuclide_name = atom.title()
            if atom.title() == "D":
                nuclide_name = "H2"
            mat_i.add_nuclide(nuclide_name , elements_i[atom] / total_atom_number, 'ao')
        mat_i.set_density("atom/cm3", float(total_atom_number) * 1e-6) # m3 -> cm3
        total_material_dict.append(mat_i)

    if download:
        mats = openmc.Materials(total_material_dict)
        if print_output:
            mats.download_cross_section_data(destination = download_location, libraries = libraries, set_OPENMC_CROSS_SECTIONS=True, particles = ['neutron'])
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    mats.download_cross_section_data(destination = download_location, libraries = libraries, set_OPENMC_CROSS_SECTIONS=True, particles = ['neutron'])
            
    os.environ["OPENMC_CROSS_SECTIONS"] = str(download_location + "/cross_sections.xml")

    return total_material_dict