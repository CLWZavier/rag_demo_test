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
from utils import get_pdf_files, get_pdf_image, update_image, get_image_count

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
            
            return f"Uploaded and extracted {len(pages)} pages", get_pdf_files()
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    
    def search_documents(self, state, query, num_results=1):
        print(f"Searching for query: {query}")
        # id = generate_uuid(state)

        print(f"===\nIndexed docs: {self.indexed_docs}\n===")
        
        if not self.indexed_docs:
            print("Please index documents first")
            return "Please index documents first", "--"
        if not query:
            print("Please enter a search query")
            return "Please enter a search query", "--"
            
        try:
            search_results = middleware.search([query])[0]
            
            doc_id = search_results[0][2]
            page_num = search_results[0][1] + 1

            print(f"Retrieved page number: {page_num}")

            img_path = f"{doc_id}/page_{page_num}.png"
            img_path = os.path.join(*img_path.split("/"))

            print(f"Retrieved image path: {img_path}")

            rag_response = rag.get_answer_from_llama(query, [img_path], model, processor)

            citation = f"Document: {doc_id}, Page: {page_num}"

            return img_path, rag_response, citation
            
        except Exception as e:
            return f"Error during search: {str(e)}", "--"

def create_ui():
    index_list = middleware.list_index()
    app = PDFSearchApp(index_list)

    textarea_css = """
    textarea, input[type="text"] {
        background-color: white !important;
    }
    """
    
    with gr.Blocks(theme='allenai/gradio-theme', css=textarea_css) as demo:
        state = gr.State(value={"user_uuid": None})

        gr.Markdown("# Colpali Milvus Multimodal RAG Demo")
        gr.Markdown("This demo showcases how to use [Colpali](https://github.com/illuin-tech/colpali) embeddings with [Milvus](https://milvus.io/) and utilizing Gemini/OpenAI multimodal RAG for pdf search and Q&A.")
        
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

                    # df = pd.DataFrame([file.replace("pages/", "", 1) for file in index_list], columns=["PDF Files"])

                    # gr.DataFrame(df)

                    # Use DataFrame with interactive=False (static table)
                    file_table = gr.DataFrame(
                        headers=["PDF Files"],
                        value=get_pdf_files(),
                        interactive=True,
                        # label="Select a PDF (Click on a row)"
                    )

                with gr.Column():
                    page_slider = gr.Slider(1, 10, value=1, step=1, label="Page")
                    image_display = gr.Image(label="PDF Page", interactive=False)
                    selected_pdf = gr.Textbox(visible=False)  # Hidden textbox to store selected PDF name

                # Function to handle row selection
                def on_select(evt: gr.SelectData):
                    pdf_name = evt.value  # Extract filename from DataFrame row
                    image_path, total_pages = get_pdf_image(pdf_name), get_image_count(pdf_name)
                    max_pages = total_pages if total_pages > 0 else 1
                    page_slider = gr.Slider(1, max_pages, value=1, step=1, label="Page")

                    return image_path, pdf_name, page_slider


                # File selection updates image and slider
                file_table.select(on_select, None, [image_display, selected_pdf, page_slider])

                # Page slider updates image
                page_slider.change(update_image, [selected_pdf, page_slider], image_display)
            
                
        
        with gr.Tab("Query"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(label="Enter query")
                    # num_results = gr.Slider(
                    #     minimum=1,
                    #     maximum=10,
                    #     value=5,
                    #     step=1,
                    #     label="Number of results"
                    # )
                    search_btn = gr.Button("Query")
                    # llm_answer = gr.Textbox(label="RAG Response", interactive=False)
                    llm_answer = gr.Markdown(label="RAG_Response", show_copy_button=True, container=True)
                with gr.Column():
                    images = gr.Image(label="Top page matching query")
                    citation = gr.Textbox(label="Citation", container=True, interactive=False)
        
        # Event handlers
        file_input.change(
            fn=app.upload_and_convert,
            inputs=[state, file_input, max_pages_input],
            outputs=[status, file_table]
        )
        
        search_btn.click(
            fn=app.search_documents,
            inputs=[state, query_input],
            outputs=[images, llm_answer, citation]
        )

        query_input.submit(
            fn=app.search_documents,
            inputs=[state, query_input],
            outputs=[images, llm_answer, citation]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(strict_cors=False)