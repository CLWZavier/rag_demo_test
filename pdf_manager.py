from pathlib import Path
from pdf2image import convert_from_path
import os
import shutil

class PdfManager:
    def __init__(self):
        pass
        
    def clear_and_recreate_dir(self, output_folder):
        print(f"Clearing output folder {output_folder}")

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

    def save_images(self, id, pdf_path, max_pages, pages: list[int] = None) -> list[str]:
        output_folder = f"pages/{id}/"
        output_folder_path = Path(output_folder)
        images = convert_from_path(pdf_path)

        # Check if the folder exists and contains images
        if output_folder_path.exists() and any(output_folder_path.glob("*.png")):
            print(f"Skipping saving images; folder '{output_folder}' already exists with images.")

        print(f"Saving images from {pdf_path} to {output_folder}. Max pages: {max_pages}")

        # self.clear_and_recreate_dir(output_folder)

        num_page_processed = 0

        for i, image in enumerate(images):
            if max_pages and num_page_processed >= max_pages:
                break

            if pages and i not in pages:
                continue

            if not output_folder_path.exists():
                os.makedirs(output_folder)

            full_save_path = f"{output_folder}/page_{i + 1}.png"

            #print(f"Saving image to {full_save_path}")

            image.save(full_save_path, "PNG")

            num_page_processed += 1

        return [f"{output_folder}/page_{i + 1}.png" for i in range(num_page_processed)]