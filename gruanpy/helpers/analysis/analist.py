from .methods.constants import Constants as CNST
from .methods.formulas import Formulas as FM
from .methods.pblh import PBLHMethods

class AnalysisManager(CNST, FM, PBLHMethods):
    """
    A class that combines methods for atmospheric analysis, including formulas and PBLH calculations.
    Inherits from Formulas and PBLHMethods to provide a comprehensive set of analytical tools.
    """
    def __init__(self):
        CNST.__init__(self)
        FM.__init__(self)
        PBLHMethods.__init__(self)