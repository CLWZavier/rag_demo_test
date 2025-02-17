import gradio as gr
import requests
import base64
import io
from PIL import Image

def query_llm(prompt, image):
    # Update this URL to match your LM Studio endpoint
    url = "http://127.0.0.1:1234/v1/completions"
    
    # Build the JSON payload with the required "prompt" field.
    payload = {"prompt": prompt}
    
    # If an image is provided, encode it in base64 and include it in the JSON payload.
    if image is not None:
        try:
            # If the image input is a file path (string)
            if isinstance(image, str):
                with open(image, "rb") as f:
                    print("Converting image file path to image")
                    image_bytes = f.read()
            else:
                # Otherwise, assume it's a PIL Image.
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                buf.seek(0)
                image_bytes = buf.read()
                
            # Convert the image bytes to a base64 string.
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            payload["image"] = image_b64
        except Exception as e:
            return f"Error processing image: {e}"
    
    try:
        # Send the JSON payload (which always contains the "prompt" field)
        response = requests.post(url, json=payload)
    except Exception as e:
        return f"Error during request: {e}"
    
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: Status code {response.status_code}\nResponse: {response.text}"

# Build the Gradio interface.
iface = gr.Interface(
    fn=query_llm,
    inputs=[
        gr.Textbox(lines=4, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Image(label="Image (optional)", type="pil")  # Use type="filepath" if you prefer a file path.
    ],
    outputs="text",
    title="Qwen2-VL LLM Query",
    description=(
        "Enter your prompt and optionally provide an image. "
        "This app sends a JSON payload (with the 'prompt' and an optional base64-encoded 'image') "
        "to your locally running LM Studio instance (qwen.qwen2-vl-7b-instruct) used in your Colpali RAG pipeline."
    )
)

if __name__ == "__main__":
    iface.launch()
