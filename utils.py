import base64
import glob
import os
import time
import csv
from constants import IMAGE_FOLDER

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

def save_logs_to_csv(model_name, query, num_results, t, log_folder="logs"):
    log_data = {
        "LLM": model_name,
        "query": query,
        "num_results": num_results,
        "elapsed": t.elapsed,
    }
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "llm_logs.csv")
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["LLM", "query", "num_results", "elapsed" ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"Elapsed time: {self.elapsed:.4f} seconds")