import base64
import glob
import os
import time
import csv
from constants.urls import IMAGE_FOLDER

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_pdf_files():
    """
    Retrieves a list of PDF files from the specified image folder.

    Returns:
        list: A list of lists, each containing the basename of a PDF file found in the image folder.
    """

    files = glob.glob(f"{IMAGE_FOLDER}/*.pdf")
    return [[os.path.basename(f)] for f in files]

def get_image_count(pdf_name):
    pdf_image_dir = os.path.join(IMAGE_FOLDER, pdf_name)
    image_files = glob.glob(f"{pdf_image_dir}/page_*.png")
    return len(image_files)

def get_pdf_image(pdf_name, page_num=1):
    if page_num < 1 or page_num > get_image_count(pdf_name):
        return None
    pdf_image_dir = os.path.join(IMAGE_FOLDER, pdf_name)
    image_file = glob.glob(f"{pdf_image_dir}/page_{page_num}.png")

    return image_file[0]

def update_image(selected_pdf, page_num):
    print(f"Updating image for {selected_pdf}, Page: {page_num}")  # Debugging log
    image_path = get_pdf_image(selected_pdf, page_num)
    print(f"image_path = {image_path}")
    return image_path

def get_pdf_file_name(file_path):
    # Extract the file name from the file path
    file_name = os.path.basename(file_path)
    return file_name

class Timer:
    def __enter__(self):
        
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"Elapsed time: {self.elapsed:.4f} seconds")