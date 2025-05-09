from .helpers.download.download_manager import DownloadManager as DM
from .helpers.read.reading_manager import ReadingManager as RM
from .helpers.grid.gridding_manager import GriddingManager as GM

class GRUANpy(DM, RM, GM):
    """
    A helper class that inherits methods from DownloadManager, ReadingManager, and GriddingManager.
    """
    pass

gruanpy = GRUANpy()