from .helpers.download.download_manager import DownloadManager as DM
from .helpers.read.reading_manager import ReadingManager as RM
from .helpers.grid.gridding_manager import GriddingManager as GM

class GRUANpy(DM, RM, GM):
    """
    A helper class that inherits methods from DownloadManager, ReadingManager, and GriddingManager.
    """
    pass

    def info(self):
        """
        Print the information about  GRUANpy.
        """
        print("GRUANpy is a toolkit for working with GRUAN data.")
        print("For more documentation regarding GRUAN data, please visit https://www.gruan.org/")
        print("For more details regarding GRUANpy, look at https://github.com/TommasoLocatelli/GRUAN_EDA")


gruanpy = GRUANpy()