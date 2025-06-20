class GD:
    """
    General Data Model for GRUAN data.
    
    This class serves as a base for all GRUAN data models, providing a structure
    to hold metadata and data attributes.
    """
    
    def __init__(self, metadata=None, data=None):
        self.metadata = metadata
        self.data = data
    
    def __call__(self):
        return self.data