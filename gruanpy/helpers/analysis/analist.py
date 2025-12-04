from .methods.constants import Constants as CNST
from .methods.formulas import Formulas as FM
from .methods.pblh import PBLHMethods
from .methods.noise import NoiseMethods
from .methods.geostatistic import GeostatisticMethods

class AnalysisManager(CNST, FM, PBLHMethods, NoiseMethods, GeostatisticMethods):
    """
    A class that combines methods for atmospheric analysis, including formulas and PBLH calculations.
    Inherits from Formulas and PBLHMethods to provide a comprehensive set of analytical tools.
    """
    def __init__(self):
        pass