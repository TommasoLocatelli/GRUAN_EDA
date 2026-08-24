from gruanpy.gdp.download import DownloadHelper
from gruanpy.gdp.read import ReadingHelper
import sys

class Helper(DownloadHelper, ReadingHelper):
    """
    A helper class that inherits methods from DownloadManager, ReadingManager, and GriddingManager.
    """
    def __init__(self):
        DownloadHelper.__init__(self)
        ReadingHelper.__init__(self)
        
    def info(self):
        """
        Print the information about GRUANpy.
        """
        print("GRUANpy is a toolkit for working with GRUAN data.")
        print("For more documentation regarding GRUAN data, please visit https://www.gruan.org/")
        print("For more details regarding GRUANpy, look at https://github.com/TommasoLocatelli/GRUAN_EDA")

helper = Helper()