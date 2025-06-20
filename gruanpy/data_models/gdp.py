from gruanpy.data_models.gd import GD

class GDP(GD):
    """
    General Data Model for GRUAN data products.
    
    Inherits from GD and adds functionality specific to data products.
    """
    
    def __init__(self, global_attrs=None, data=None, variables_attrs=None, metadata=None):
        super().__init__(metadata=metadata, data=data)
        self.global_attrs = global_attrs
        self.variables_attrs = variables_attrs