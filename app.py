import gradio as gr
import tempfile
import os
import fitz  # PyMuPDF
import uuid
import torch
import pandas as pd

from transformers import (
    MllamaForConditionalGeneration, 
    AutoProcessor,
    BitsAndBytesConfig
)
from middleware import Middleware
from rag import Rag
from pathlib import Path
from pymilvus import connections, utility
from utils import get_pdf_files, get_pdf_image, update_image, get_image_count, Timer, save_logs_to_csv

rag = Rag()

print("Loading generation model...")

# Quantization config for generative model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

print(f"model memory = {model.get_memory_footprint()}")

middleware = Middleware()

def generate_uuid(state):
    # Check if UUID already exists in session state
    
    if state["user_uuid"] is None:
        # Generate a new UUID if not already set
        state["user_uuid"] = str(uuid.uuid4())

    return state["user_uuid"]

def get_pdf_file_name(file_path):
    # Extract the file name from the file path
    file_name = os.path.basename(file_path)
    return file_name

def get_id_from_folder(base_folder: str) -> list[str]:
    """
    Retrieves the list of IDs (folder names) in the base folder.

    :param base_folder: The root folder containing ID-based subfolders.
    :return: A list of IDs corresponding to existing subfolders.
    """
    base_path = Path(base_folder)
    
    if not base_path.exists():
        print(f"Base folder '{base_folder}' does not exist.")
        return []

    # List all subfolders and extract their names
    ids = [folder.name for folder in base_path.iterdir() if folder.is_dir()]
    if ids:
        return ids
    return None

class PDFSearchApp:
    def __init__(self, index_list):
        self.indexed_docs = {}
        self.model = model
        self.processor = processor

        if index_list:
            for i in index_list:
                self.indexed_docs[i] = True
        else:
            self.indexed_docs = {}

    def upload_and_convert(self, state, file, max_pages):
        # id = generate_uuid(state)

        if file is None:
            return "No file uploaded", get_pdf_files()

        file_name = get_pdf_file_name(file.name)

        id = file_name
        state["id"] = id

        if self.indexed_docs and id in self.indexed_docs:
            return f"Document {file_name} already indexed."

        print(f"Uploading file: {file_name}, id: {file_name}")

        try:
            pages = middleware.index(pdf_path=file.name, id=file_name, max_pages=max_pages)

            self.indexed_docs[id] = True
            
            return f"Uploaded and extracted {len(pages)} pages"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    
    def search_documents(self, state, query, num_results=3):
        if not self.indexed_docs:
            print("Please index documents first")
            return "Please index documents first", "--"
        if not query:
            print("Please enter a search query")
            return "Please enter a search query", "--"
            
        try:
            img_paths = []
            doc_ids = []
            page_nums = []
            search_results = middleware.search([query], num_results)[0]

            try:
                for i in range(len(search_results)):
                    doc_id = search_results[i][2]
                    page_num = search_results[i][1] + 1

                    print(f"Retrieved page number: {page_num}")

                    img_path = f"{doc_id}/page_{page_num}.png"
                    img_path = os.path.join(*img_path.split("/"))

                    print(f"Retrieved image path: {img_path}")
                    img_paths.append(img_path)
                    doc_ids.append(doc_id)
                    page_nums.append(page_num)
            except Exception as e:
                return f"Error in for loop: {str(e)}"

            with Timer() as t:
                rag_response = rag.get_answer_from_llama(query, img_paths, model, processor, num_results)
            
            save_logs_to_csv(model_id, query, num_results, t, log_folder="logs")

            gallery_data = [(img_paths[i], f"Document: {doc_ids[i]}, Page: {page_nums[i]}") for i in range(len(img_paths))]
            print(f"gallery data = {gallery_data}")
            return gallery_data, rag_response

            
        except Exception as e:
            return f"Error during search: {str(e)}", "--"
    
def delete_pdf(selected_pdf):
    """Deletes a PDF file from local storage and Milvus."""
    if not selected_pdf:
        return "No PDF selected.", None, None, None
    
    pdf_path = os.path.join("pages", selected_pdf)

    try:
        # Delete from Milvus
        middleware.milvus_manager.delete_doc("pages/" + selected_pdf)

        # Only do this once deleted from milvus
        if os.path.exists(pdf_path):
            import shutil
            shutil.rmtree(pdf_path)

        # Get updated list of PDFs
        updated_files = get_pdf_files()
        
        # Create updated choices for dropdown
        updated_choices = [pdf for sublist in updated_files for pdf in sublist]
        
        # Create updated DataFrame
        updated_df = gr.DataFrame(value=updated_files, headers=["PDF Files"])
        
        # Create a new dropdown component with updated choices
        updated_dropdown = gr.Dropdown(
            choices=updated_choices,
            label="Select PDF to Delete",
            filterable=True,
            container=True,
            scale=6,
            value=None  # Reset the selected value
        )

        return f"Deleted {selected_pdf} successfully.", updated_dropdown, updated_df, None

    except Exception as e:
        return f"Error deleting {selected_pdf}: {str(e)}", None, None, None

def create_ui():
    index_list = middleware.list_index()
    app = PDFSearchApp(index_list)

    css = """
    textarea, input[type="text"] {
        background-color: white !important;
    }

    .wrap.svelte-1hfxrpf.svelte-1hfxrpf {
        background-color: white !important;
    }
    
    .wrap-inner.svelte-1hfxrpf.svelte-1hfxrpf {
        background-color: white !important;
    }
    """
    
    with gr.Blocks(theme='allenai/gradio-theme', css=css) as demo:
        state = gr.State(value={"user_uuid": None})

        gr.Markdown("# Colpali Milvus Multimodal RAG Demo")
        
        with gr.Tab("Upload PDF"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload PDF", interactive=True)
                    
                    max_pages_input = gr.Slider(
                        minimum=1,
                        maximum=1000,
                        value=100,
                        step=10,
                        label="Max pages to extract and index"
                    )
                    
                    status = gr.Textbox(label="Indexing Status", interactive=False)

                    file_table = gr.DataFrame(
                        headers=["PDF Files"],
                        value=get_pdf_files(),
                        interactive=False,
                    )
                    
                    with gr.Row():
                        current_files = [pdf for sublist in get_pdf_files() for pdf in sublist]
                        pdf_dropdown = gr.Dropdown(
                            choices=current_files,
                            value=None,  # Set initial value to None
                            label="Select PDF to Delete",
                            filterable=True,
                            container=True,
                            scale=6,
                        )
                        delete_btn = gr.Button("Delete")

                with gr.Column():
                    page_slider = gr.Slider(1, 10, value=1, step=1, label="Page")
                    image_display = gr.Image(label="PDF Page", interactive=False)
                    selected_pdf = gr.Textbox(visible=False)

                def on_select(evt: gr.SelectData):
                    pdf_name = evt.value
                    image_path, total_pages = get_pdf_image(pdf_name), get_image_count(pdf_name)
                    max_pages = total_pages if total_pages > 0 else 1
                    return image_path, pdf_name, gr.Slider(1, max_pages, value=1, step=1, label="Page")

                # Update both dropdown and file table when a file is uploaded
                def update_components(status_msg):
                    current_files = [pdf for sublist in get_pdf_files() for pdf in sublist]
                    return (
                        status_msg,
                        get_pdf_files(),
                        gr.Dropdown(
                            choices=current_files,
                            value=None,
                            label="Select PDF to Delete",
                            filterable=True,
                            container=True,
                            scale=6,
                        )
                    )

                file_table.select(on_select, None, [image_display, selected_pdf, page_slider])
                page_slider.change(update_image, [selected_pdf, page_slider], image_display)
        
        with gr.Tab("Query"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(label="Enter query")
                    num_results_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of results"
                    )
                    search_btn = gr.Button("Query")
                    llm_answer = gr.Markdown(label="RAG_Response", show_copy_button=True, container=True)
                with gr.Column():
                    images = gr.Gallery(label="Top pages matching query", object_fit="contain")
        
        # Event handlers
        file_input.change(
            fn=app.upload_and_convert,
            inputs=[state, file_input, max_pages_input],
            outputs=[status]
        ).then(
            fn=update_components,
            inputs=[status],
            outputs=[status, file_table, pdf_dropdown]
        )
        
        delete_btn.click(
            fn=delete_pdf,
            inputs=pdf_dropdown,
            outputs=[status, pdf_dropdown, file_table, image_display]
        )

        # file_input.change(
        #     fn=app.upload_and_convert,
        #     inputs=[state, file_input, max_pages_input],
        #     outputs=[status, file_table]
        # )
        
        search_btn.click(
            fn=app.search_documents,
            inputs=[state, query_input, num_results_slider],
            outputs=[images, llm_answer]
        )

        query_input.submit(
            fn=app.search_documents,
            inputs=[state, query_input, num_results_slider],
            outputs=[images, llm_answer]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(strict_cors=False, share=True)