"""
This module provides a DownloadManager (DM) class for managing Gruan data product (GDP) downloads through an FTP server.
As of 2025, GRUAN has not yet implemented an authentication system for complete data dissemination.
To download the data, you need to register on the GRUAN website and manually download the GDP of interest.
Alternatively, the DM class helps in downloading some examples of GDP made available by an FTP server of the 
National Oceanic and Atmospheric Administration (NOAA).
The class exploit the ftplib library to connect to the FTP server and download files and can be used in principle to download other types of binary file from others FTP servers.

See an example of usage to explore and download GDP from the NOAA FTP server in "code_examples\download_gdp.py".

Attributes:
    ftp_url (str): The URL of the FTP server. Defaults to "ftp.ncdc.noaa.gov".
    download_folder (str): The local folder where downloaded files will be stored. Defaults to "gdp".

Methods:
    __init__(ftp_url="ftp.ncdc.noaa.gov", download_folder="gdp"):
        Initializes the DM with the specified FTP URL and download folder.

    search(ftp_dir_path):
        Searches for files in the specified FTP directory.
        Args:
            ftp_dir_path (str): The path to the directory on the FTP server.
        Returns:
            list: A list of filenames in the specified FTP directory.

    download(ftp_dir_path, filename):
        Downloads a file from the specified FTP directory to the local download folder.
        Args:
            ftp_dir_path (str): The path to the directory on the FTP server.
            filename (str): The name of the file to download.
        Returns:
            None
"""

from ftplib import FTP
import os

class DownloadManager:
    def __init__(self, ftp_url="ftp.ncdc.noaa.gov", download_folder="gdp"):
        self.ftp_url=ftp_url
        self.download_folder=download_folder

    def search(self, ftp_dir_path=r'pub/data/gruan/processing'):

        ftp=FTP(self.ftp_url)
        ftp.login()
        ftp.cwd(ftp_dir_path)
        files = ftp.nlst()
        ftp.quit()

        return files
    
    def download(self, ftp_dir_path, filename):

        ftp = FTP(self.ftp_url)
        ftp.login()
        ftp.cwd(ftp_dir_path)
        
        os.makedirs(self.download_folder, exist_ok=True)
        local_file_path = os.path.join(self.download_folder, filename)
        with open(local_file_path, 'wb') as local_file:
            ftp.retrbinary(f"RETR {filename}", local_file.write)
        
        ftp.quit()

