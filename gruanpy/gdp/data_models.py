class NETCDF:
    """
    General Data Model for NETCDF.
    """
    
    def __init__(self, global_attrs=None, data=None, variables_attrs=None):
        self.global_attrs = global_attrs
        self.data = data
        self.variables_attrs = variables_attrs


from .quality_check import (
    missing_data,
    physics_constraint,
    detect_outliers,
    altitude_drops
)


class GDP(NETCDF):
    """
    General Data Model for GRUAN data products.
    Automatically performs QC checks at initialization.
    """

    def __init__(self, global_attrs=None, data=None, variables_attrs=None):
        super().__init__(global_attrs, data, variables_attrs)

        # dictionary to store QC outputs
        self.qc_results = {}

        # run QC immediately
        self.apply_quality_checks()

    def apply_quality_checks(self):

        if self.data is None:
            self.qc_results = {}
            return self.qc_results

        self.qc_results["missing_data"] = missing_data(self.data)

        self.qc_results["physics_constraint"] = physics_constraint(self.data)

        self.qc_results["detect_outliers"] = detect_outliers(self.data)

        self.qc_results["altitude_drops"] = altitude_drops(self.data)

        return self.qc_results
