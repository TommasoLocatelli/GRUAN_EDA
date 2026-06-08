from .methods.constants import Constants as CNST
from .methods.formulas import Formulas as FM
from .methods.pblh import PBLHMethods
from .methods.noise import NoiseMethods

class AnalysisManager(CNST, FM, PBLHMethods, NoiseMethods):
    """
    A class that combines methods for atmospheric analysis, including formulas and PBLH calculations.
    Inherits from Formulas and PBLHMethods to provide a comprehensive set of analytical tools.
    """
    def __init__(self):
        CNST.__init__(self)
        FM.__init__(self)
        PBLHMethods.__init__(self)
        NoiseMethods.__init__(self)