import middleware
import gradio as gr
from app import PDFSearchApp
from utils import get_pdf_files, get_pdf_image, update_image, get_image_count

def create_ui():
    index_list = middleware.list_index()
    app = PDFSearchApp(index_list)

    textarea_css = """
    textarea, input[type="text"] {
        background-color: white !important;
    }
    """
    
    with gr.Blocks(theme='allenai/gradio-theme', css=textarea_css) as demo:
        state = gr.State(value={"user_uuid": None, "num_results": 3})

        gr.Markdown("# Colpali Milvus Multimodal RAG Demo")
        # gr.Markdown("This demo showcases how to use [Colpali](https://github.com/illuin-tech/colpali) embeddings with [Milvus](https://milvus.io/) and utilizing Gemini/OpenAI multimodal RAG for pdf search and Q&A.")
        
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
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of results"
                    )
                    search_btn = gr.Button("Query")
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