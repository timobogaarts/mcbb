'''Blanket definitions: layers, element atom densities and thicknesses. No openmc import, so the definitions
can be used (and saved / loaded as JSON) without openmc installed.'''
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class BlanketLayer:
    name     : str
    elements : Dict


@dataclass
class Blanket(ABC):
    layers : List[BlanketLayer]


    
    @property 
    @abstractmethod
    def average_thicknesses(self):
        '''
        Average thicknesses of the layers in the blanket. 
        Should be a list of the same length as the number of layers, where each entry is the average thickness of that layer in meters. 
        The first layer is expected to be the plasma, and the rest are the blanket layers.'''
        ...

    @property     
    def average_thickness(self):
        '''
        Average thickness of the entire blanket, in meters. 
        '''
        return sum(self.average_thicknesses)


    @property
    def n_layers(self):
        return len(self.layers)
    
    @property
    def layer_names(self):
        return [layer.name for layer in self.layers]


@dataclass
class BlanketLayer1D(BlanketLayer):
    thickness : float  # in meters

@dataclass
class Blanket1D(Blanket):
    size_yz : float  # in meters
    prism_boundary_type : str = 'periodic'  # e.g., 'reflective', 'vacuum', etc.
    plasma_boundary_type : str = 'vacuum'

    @property     
    def average_thicknesses(self):
        return self.thicknesses

    @property
    def thicknesses(self):
        return [layer.thickness for layer in self.layers]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d : dict) -> "Blanket1D":
        d = dict(d)
        return cls([BlanketLayer1D(**layer) for layer in d.pop("layers")], **d)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path) -> "Blanket1D":
        return cls.from_dict(json.loads(Path(path).read_text()))
