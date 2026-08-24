class GDP():
    """
    General Data Model for GRUAN data products.
    """
    
    def __init__(self, global_attrs=None, data=None, variables_attrs=None):
        self.global_attrs = global_attrs
        self.data = data
        self.variables_attrs = variables_attrs