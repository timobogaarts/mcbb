import os 
import contextlib
from typing import List
from .blanket_base import Blanket
def create_openmc_ce_materials(blanket : Blanket, download=False, download_location = "",  libraries  : List[str] = ["FENDL-3.1d"], print_output = False) -> dict:
    '''
    Function to create continuous-energy materials for OpenMC from a dictionary of materials

    It also sets the OPENMC_CROSS_SECTIONS variable.

    Parameters
    ----------
    blanket : Blanket
        A Blanket object containing the materials of the blanket. Density assumed in m3
    download : bool, optional
        Whether to download the cross section data, by default False
    download_location : str, optional
        The location to download the cross section data to, by default ""
    libraries : List[str], optional
        The libraries to download, by default ["FENDL-3.1d"]
    print_output : bool, optional
        Whether to print the output of the download, by default False
    
    Returns
    -------    
    total_material_list : List
        A list containing the OpenMC materials    

    '''
    import openmc
    import openmc_data_downloader

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