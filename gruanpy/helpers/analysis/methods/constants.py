# Constants used in various meteorological calculations
# Values are in SI units unless otherwise specified

class Constants:
    def __init__(self):
        self.R_STAR = 8.3145  # J/(mol·K) Universal gas constant
        self.R_DRY_AIR = 287.05  # J/(kg·K) Specific gas constant for dry air
        self.R_WATER_VAPOR = 461.495  # J/(kg·K) Specific gas
        self.M_DRY_AIR = 0.0289647  # kg/mol Molar mass of dry air
        self.M_WATER_VAPOR = 0.01801528  # kg/mol Molar mass of water
        self.EPSILON = 0.622  # Dimensionless Ratio of the molar masses of water vapor to dry air
        self.C_P_DRY_AIR = 1005.7  # J/(kg·K) Specific heat capacity of dry air at constant pressure
        self.Poisson_exponent = self.R_DRY_AIR / self.C_P_DRY_AIR  # Dimensionless Poisson exponent for dry air
        self.DRY_ADIABATIC_LAPSE_RATE = 9.8  # K/km Dry adiabatic lapse rate
        self.G0 = 9.80665  # m/s² Averaged gravity
        self.RIC = 0.25  # Dimensionless Richardson critical number for turbulence