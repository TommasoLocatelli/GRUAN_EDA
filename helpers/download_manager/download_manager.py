from helpers.download_manager.utils import get_image_from_url, get_image_from_class, get_pdf_from_url
import shutil

class DownloadManager:
    def __init__(self):
        self.download_folder="download\\"

    def download_image_from_url(self,
            image_url = "https://www.gruan.org/gruan/_processed_/3/b/csm_GRUAN-map-web_e665c52013.png",
            downloaded_image_name="GRUAN_map_web.png"):
        
        get_image_from_url(image_url,
                self.download_folder+downloaded_image_name)

    def download_image_from_class(self,
            web_page_url="https://www.gruan.org/", 
            image_class_name="img-fluid", 
            downloaded_image_name="GRUAN_logo.png"):
        
        get_image_from_class(web_page_url,
                image_class_name,
                self.download_folder+downloaded_image_name)
    
    def download_pdf_from_url(self,
            pdf_url="https://www.gruan.org/gruan/editor/documents/gruan/GRUAN-TN-1_NewRS_v1.0.pdf",
            downloaded_pdf_name="GRUAN-TN-1_NewRS_v1_0.pdf"):
        
        get_pdf_from_url(pdf_url, self.download_folder+downloaded_pdf_name)

    def move_file(self, source_path, destination_path):
        try:
            shutil.move(source_path, destination_path)
            print(f"Moved file from {source_path} to {destination_path}")
        except Exception as e:
            print(f"Error moving file: {e}")     
