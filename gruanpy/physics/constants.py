# Constants used in various calculations
R_STAR = 8.3145  # J/(mol·K) Universal gas constant
R_DRY_AIR = 287.05  # J/(kg·K) Specific gas constant for dry air
R_WATER_VAPOR = 461.495  # J/(kg·K) Specific gas
M_DRY_AIR = 0.0289647  # kg/mol Molar mass of dry air
M_WATER_VAPOR = 0.01801528  # kg/mol Molar mass of water
EPSILON = 0.622  # Dimensionless Ratio of the molar masses of water vapor to dry air
C_P_DRY_AIR = 1005.7  # J/(kg·K) Specific heat capacity of dry air at constant pressure
Poisson_exponent = R_DRY_AIR / C_P_DRY_AIR  # Dimensionless Poisson exponent for dry air
DRY_ADIABATIC_LAPSE_RATE = 9.8  # K/km Dry adiabatic lapse rate
G0 = 9.80665  # m/s² Averaged gravity
RIC = 0.25  # Dimensionless Richardson critical number for turbulence
p0 = 1000.0  # hPa Reference pressure for potential temperature calculations