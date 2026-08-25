"""
Functions for exploring and downloading GRUAN Data Products (GDP)
from the NOAA FTP server.

As of 2025, GRUAN does not provide authenticated programmatic access
to full GDP datasets. These utilities allow downloading example GDP
files made publicly available via NOAA's FTP server.

Example usage in: apps/download_gdp.py
"""

from ftplib import FTP
import os


DEFAULT_FTP_URL = "ftp.ncdc.noaa.gov"
DEFAULT_DOWNLOAD_FOLDER = "data"


def search_gdp(ftp_dir_path="pub/data/gruan/processing", ftp_url=DEFAULT_FTP_URL):
    """Return a list of files/directories inside an FTP directory."""
    ftp = FTP(ftp_url, timeout=30)
    ftp.login()
    ftp.cwd(ftp_dir_path)
    items = ftp.nlst()
    ftp.quit()
    return items


def download_gdp(ftp_dir_path, file_name, ftp_url=DEFAULT_FTP_URL,
                 download_folder=DEFAULT_DOWNLOAD_FOLDER):
    """Download a file from the FTP server into a local folder."""
    ftp = FTP(ftp_url, timeout=30)
    ftp.login()
    ftp.cwd(ftp_dir_path)

    os.makedirs(download_folder, exist_ok=True)
    local_path = os.path.join(download_folder, file_name)

    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {file_name}", f.write)

    ftp.quit()
    return local_path


def exec_cds_request(api_request):
    """
    Execute a Copernicus CDS API request string.

    The request must begin with 'import cdsapi'.
    """
    assert isinstance(api_request, str), "api_request must be a string"
    api_request = api_request.lstrip()
    assert api_request.startswith("import cdsapi"), \
        "api_request must start with 'import cdsapi'"
    exec(api_request)
