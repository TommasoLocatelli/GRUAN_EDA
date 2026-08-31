import pytz
from datetime import datetime

SITE_TIMEZONES = {
    "LIN": "Europe/Berlin",
    "HKO": "Asia/Hong_Kong",
    "LAU": "Pacific/Auckland",
    "POT": "Europe/Rome",
}

def get_utc_time(gdp):
    # Scheduled or standard date & time of measurement [ISO 8601]
    return gdp.global_attrs.loc[
        gdp.global_attrs['Attribute'] == 'g.Measurement.StandardTime',
        'Value'
    ].values[0]

def get_time_of_day(gdp):
    # Term to describe the time of day of flight (daytime, nighttime, twilight)
    return gdp.global_attrs.loc[
        gdp.global_attrs["Attribute"] == "g.Measurement.TimeOfDay",
        "Value"
    ].values[0]

def to_local_time(utc_timestamp, site_key):
    tz = pytz.timezone(SITE_TIMEZONES[site_key])
    utc_dt = datetime.fromisoformat(utc_timestamp.replace("Z", "+00:00"))
    return utc_dt.astimezone(tz)

def classify_season(dt, site_key):
    m = dt.month

    if site_key == "LAU":  # southern hemisphere
        if m in (12, 1, 2): return "summer"
        if m in (3, 4, 5): return "autumn"
        if m in (6, 7, 8): return "winter"
        return "spring"

    # northern hemisphere
    if m in (12, 1, 2): return "winter"
    if m in (3, 4, 5): return "spring"
    if m in (6, 7, 8): return "summer"
    return "autumn"