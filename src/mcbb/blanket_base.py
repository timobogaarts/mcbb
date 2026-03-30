from dataclasses import dataclass, fields
from typing import Dict, List
from abc import ABC, abstractmethod
import openmc
from functools import cached_property
from typing import Union, Literal, Iterable
import numpy as np
import copy


def _core_dict(self):
    '''
    Helper for a deepcopy without any cached properties: just returns the fields.
    '''
    return {
        f.name: getattr(self, f.name)
        for f in fields(self)
    }

@dataclass
class BlanketLayer:
    name     : str
    elements : Dict


@dataclass
class Blanket(ABC):
    layers : List[BlanketLayer]


    
    @property 
    @abstractmethod
    def average_thickness(self):
        ...

    @property
    def n_layers(self):
        return len(self.layers)


@dataclass
class OpenMCSettingsBlanket:
    download         : bool
    download_location : str
    batches : int 
    samples : int 
    verbosity : int = 1
    seed     : int = None
    weight_windows : openmc.WeightWindows  = None

@dataclass
class OpenMCBlanketSimulation(ABC):
    blanket           :  Blanket
    settings_blanket  : OpenMCSettingsBlanket

    @cached_property
    def materials(self):
        from .materials import create_openmc_ce_materials
        return openmc.Materials(create_openmc_ce_materials(self.blanket, self.settings_blanket.download, self.settings_blanket.download_location))        
    
    @cached_property
    @abstractmethod
    def geometry(self):
        ...

    @cached_property
    @abstractmethod
    def tallies(self):
        ...

    @cached_property
    @abstractmethod
    def source(self):
        ...

    @abstractmethod
    def get_tally_results(self, sp_file : Union[str, None] = None, quantities : List[Literal["mean", "std_dev", "rel_err"]] = ["mean"]):
        ...

    @cached_property
    def model(self):
        return openmc.Model(self.geometry, self.materials, self.settings, self.tallies)

    @cached_property
    def settings(self):
        from .openmc_base_functions import create_openmc_settings
        
        return create_openmc_settings(
            batches        = self.settings_blanket.batches,
            samples        = self.settings_blanket.samples,
            source         = self.source,
            verbosity      = self.settings_blanket.verbosity,
            seed           = self.settings_blanket.seed,
            weight_windows = self.settings_blanket.weight_windows
        )
    
    def with_weight_windows(self, weight_windows : openmc.WeightWindows):
        '''
        Create a new object that has weight windows in the settings.
        The rest of the geometry, materials, and tallies are unchanged.

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
        new = type(self)(**_core_dict(self))
        new.settings_blanket.weight_windows = weight_windows
        
        return new
    
    def run(self, statepoint_file : str = None):
        import shutil
        '''
        Function to run the OpenMC simulation
        Returns
        -------
        sp_file : str
            The path to the OpenMC statepoint file
        -------
        '''
        import os 

        if (statepoint_file != None) and (os.path.exists(statepoint_file)):
            print("Cached at statepoint file: " + statepoint_file)
            self.sp_file = statepoint_file
            return statepoint_file
        model = openmc.Model(self.geometry, self.materials, self.settings, self.tallies)
        sp_file = model.run(threads=12)
        
        if statepoint_file is not None:
            shutil.move(sp_file, statepoint_file)
            sp_file = statepoint_file
        
        self.sp_file = sp_file
        return sp_file

    def _setup_mgxs_lib(self, egroups : Iterable[float], legendre_order : int, extra_types  : List[str] = ['(n,Xt)', 'heating']):
        from .openmc_base_functions import create_mgxs_lib
        mgxs_lib = create_mgxs_lib(
            openmc_geometry = self.geometry,
            egroups         = egroups,
            legendre_order  = legendre_order,
            extra_types     = extra_types
        )
        return mgxs_lib
    
    def create_mgxs_library(self,  energy_groups : Iterable[float], legendre_order : int, extra_types : List[str] = ['(n,Xt)', 'heating']):
        '''
        Function to create a multi-group cross-section library for the OpenMC simulation

        Runs the simulation using the specified settings as well; 
        results are available in the same way using get_tally_results.

        Parameters
        ----------
        legendre_order : int
            The Legendre order for the multi-group cross-section library
        energy_groups : Iterable[float]
            The energy group boundaries for the multi-group cross-section library, ordered from low to high
        extra_types : List[str]
            A list of extra multi-group cross-section types to include in the library (default is ['(n,Xt)', 'heating'])
        
        Returns
        -------
        mgxs_lib : openmc.mgxs.Library
            The OpenMC multi-group cross-section library object for the breeding blanket, loaded with the data
        
        -------
        '''
        mgxs_lib = self._setup_mgxs_lib(energy_groups, legendre_order, extra_types)

        tallies_copy = self.tallies[:]
        tallies_omc = openmc.Tallies(tallies_copy)
        mgxs_lib.add_to_tallies(tallies_omc, merge=True)                
        model = openmc.Model(self.geometry, self.materials, self.settings, tallies_omc)
        
        sp_file = model.run(threads=12)

        with openmc.StatePoint(sp_file) as sp:
            mgxs_lib.load_from_statepoint(sp)
        
        self.sp_file = sp_file

        return mgxs_lib
      
    def plot_geometry(self,  basis, slice_coord):
        plots = openmc.Plot.from_geometry(self.geometry, basis = basis, slice_coord=slice_coord)
        #plots.color_by = 'material'
        self.materials.export_to_xml()
        self.geometry.export_to_xml()
        openmc.plot_inline(plots)
        


    
    
