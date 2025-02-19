import gradio as gr
import os
import uuid
import torch
import pandas as pd
import gc

from transformers import (
    MllamaForConditionalGeneration, 
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration
)
from middleware import Middleware
from rag import Rag
from pathlib import Path
from pymilvus import connections, utility
from utils import (
    get_pdf_files, 
    get_pdf_image, 
    update_image, 
    get_image_count, 
    Timer, 
    save_logs_to_csv, 
    get_pdf_file_name
    )

rag = Rag()

print("Loading generation model...")

# Quantization config for generative model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

global_model_id = None

# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model_id = "Qwen/Qwen2-VL-7B-Instruct"
# model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     quantization_config=bnb_config,
#     device_map="auto",
# )
# processor = AutoProcessor.from_pretrained(model_id)

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     quantization_config=bnb_config,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# processor = AutoProcessor.from_pretrained(model_id)

# print(f"model memory = {model.get_memory_footprint()}")

middleware = Middleware()

## For reference
# def generate_uuid(state):
#     # Check if UUID already exists in session state
    
#     if state["user_uuid"] is None:
#         # Generate a new UUID if not already set
#         state["user_uuid"] = str(uuid.uuid4())

#     return state["user_uuid"]

class PDFSearchApp:
    def __init__(self, index_list):
        self.indexed_docs = {}
        # self.model = model
        # self.processor = processor

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
    
    
    def search_documents(self, model_processor, query, num_results=3):
        if not self.indexed_docs:
            return None, "Please index documents first"
        if not query:
            return None, "Please enter a search query"
        if not model_processor or not model_processor.get("model"):
            return None, "No model loaded - select a model first"
            
        try:
            img_paths = []
            doc_ids = []
            page_nums = []
            search_results = middleware.search([query], num_results)[0]
            model = model_processor["model"]
            processor = model_processor["processor"]

            try:
                for i in range(len(search_results)):
                    doc_id = search_results[i][2]
                    page_num = search_results[i][1] + 1

                    print(f"Retrieved page number: {page_num}")

                    # Construct the full path to the image
                    img_path = f"{doc_id}/page_{page_num}.png"
                    img_path = os.path.join(*img_path.split("/"))
                    
                    # Verify the image exists
                    if not os.path.exists(img_path):
                        print(f"Warning: Image not found at {img_path}")
                        continue

                    print(f"Retrieved image path: {img_path}")
                    img_paths.append(img_path)
                    doc_ids.append(doc_id)
                    page_nums.append(page_num)

                if not img_paths:
                    return None, "No matching images found"

                print(f"model.config.model_type = {model.config.model_type}")

                with Timer() as t:
                    if "mllama" in model.config.model_type:
                        rag_response = Rag.get_answer_from_llama(query, img_paths, model, processor, num_results)
                    elif "qwen2_vl" in model.config.model_type:
                        rag_response = Rag.get_answer_from_qwen(query, img_paths, model, processor, num_results)
                
                print("Logging data...")
                save_logs_to_csv(global_model_id, query, num_results, t, log_folder="logs")

                # Format gallery data as list of (image_path, caption) tuples
                gallery_data = []
                for i in range(len(img_paths)):
                    try:
                        # Read the image to verify it's valid
                        caption = f"Document: {os.path.basename(doc_ids[i])}, Page: {page_nums[i]}"
                        gallery_data.append((img_paths[i], caption))
                    except Exception as e:
                        print(f"Error processing image {img_paths[i]}: {str(e)}")
                        continue

                if not gallery_data:
                    return None, "No valid images found"

                return gallery_data, rag_response

            except Exception as e:
                print(f"Error in processing results: {str(e)}")
                return None, f"Error processing results: {str(e)}"
                
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return None, f"Error during search: {str(e)}"

def load_model(model_id):
    """Loads model and processor with memory cleanup"""
    print(f"Loading {model_id}...")
    torch.cuda.empty_cache()
    
    if "Qwen" in model_id:
        model_cls = Qwen2VLForConditionalGeneration
        kwargs = {"attn_implementation": "flash_attention_2"}
    else:
        model_cls = MllamaForConditionalGeneration
        kwargs = {}
    
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        **kwargs
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return {"model": model, "processor": processor}

def unload_model(current_models):
    """Clears model from memory"""
    if current_models and current_models["model"]:
        del current_models["model"]
        del current_models["processor"]
        torch.cuda.empty_cache()
        gc.collect()
    return {"model": None, "processor": None}

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

        model_processor = gr.State({"model": None, "processor": None})
        current_model_id = gr.State(None)

        gr.Markdown("# Colpali Milvus Multimodal RAG Demo")
        
        # Tab for choosing LLM and settings its hyperparamters
        with gr.Tab("LLM Settings"):
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=["Qwen/Qwen2-VL-7B-Instruct", "meta-llama/Llama-3.2-11B-Vision-Instruct"],
                        value=current_model_id,
                        label="Select LLM model to use:",
                        filterable=True,
                        container=True,
                        scale=7
                    )
                    select_model_btn = gr.Button("Use model", scale=1)

        # Tab for ingesting documents
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
                def update_ingestion_components(status_msg):
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
                
                def update_model(model_id, current_models):
                    global global_model_id
                    global_model_id = model_id
                    if model_id == current_model_id.value:
                        return current_models
                    
                    # Unload existing model
                    unload_model(current_models)
                    
                    # Load new model
                    new_models = load_model(model_id)
                    current_model_id.value = model_id
                    return new_models

                file_table.select(on_select, None, [image_display, selected_pdf, page_slider])
                page_slider.change(update_image, [selected_pdf, page_slider], image_display)
        
        # Tab for search
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
        
        # Event handlers - Tab for choosing LLM and settings its hyperparamters
        select_model_btn.click(
            fn=update_model,
            inputs=[model_dropdown, model_processor],
            outputs=[model_processor]
        )

        # Event handlers - Tab for ingesting documents
        file_input.change(
            fn=app.upload_and_convert,
            inputs=[state, file_input, max_pages_input],
            outputs=[status]
        ).then(
            fn=update_ingestion_components,
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
        
        # Event handlers - Tab for search
        search_btn.click(
            fn=app.search_documents,
            inputs=[model_processor, query_input, num_results_slider],
            outputs=[images, llm_answer]
        )

        query_input.submit(
            fn=app.search_documents,
            inputs=[model_processor, query_input, num_results_slider],
            outputs=[images, llm_answer]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(strict_cors=False, share=True)