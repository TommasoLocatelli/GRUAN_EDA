from .helpers.download.download_manager import DownloadManager
from .helpers.read.reading_manager import ReadingManager
from .helpers.grid.gridding_manager import GriddingManager
from .helpers.analysis.analist import AnalysisManager
import sys

class GRUANpy(DownloadManager, ReadingManager, GriddingManager, AnalysisManager):
    """
    A helper class that inherits methods from DownloadManager, ReadingManager, and GriddingManager.
    """
    def __init__(self):
        DownloadManager.__init__(self)
        ReadingManager.__init__(self)
        GriddingManager.__init__(self)
        AnalysisManager.__init__(self)
        
    def info(self):
        """
        Print the information about GRUANpy.
        """
        print("GRUANpy is a toolkit for working with GRUAN data.")
        print("For more documentation regarding GRUAN data, please visit https://www.gruan.org/")
        print("For more details regarding GRUANpy, look at https://github.com/TommasoLocatelli/GRUAN_EDA")

gp = GRUANpy()
sys.modules[__name__] = gp