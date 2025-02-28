from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
from io import BytesIO
import requests
from urllib.parse import urljoin

def get_image_from_url(image_url, downloaded_image_name):
    
    # Send a request to the image URL
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the image from the response content
        image = Image.open(BytesIO(response.content))
        
        # Save the image locally as PNG to preserve quality
        image.save(downloaded_image_name)
        print("Image downloaded successfully")
    else:
        print("Failed to download the image")

def get_image_from_class(web_page_url, 
              image_class_name, 
              downloaded_image_name):  # Save as PNG
    # Set up the WebDriver (this example uses Chrome)
    driver = webdriver.Chrome()

    # Open the web page
    driver.get(web_page_url)  # Replace with the actual URL of the web page

    # Wait for the image element to be present
    wait = WebDriverWait(driver, 10)
    image_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, image_class_name)))

    # Get the image URL from the image element's src attribute
    image_url = image_element.get_attribute("src")

    # Construct the full URL if the image URL is relative
    full_image_url = urljoin(web_page_url, image_url)  # Use the web page base URL

    # Send a request to the full image URL
    response = requests.get(full_image_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the image from the response content
        image = Image.open(BytesIO(response.content))
        
        # Save the image locally as PNG to preserve quality
        image.save(downloaded_image_name)
        print("Image downloaded successfully")
    else:
        print("Failed to download the image")

    # Close the WebDriver
    driver.quit()

def get_pdf_from_url(pdf_url, downloaded_pdf_name):
    # Send a request to the URL
    response = requests.get(pdf_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the PDF file locally
        with open(downloaded_pdf_name, "wb") as file:
            file.write(response.content)
        print("PDF downloaded successfully")
    else:
        print("Failed to download the PDF")
