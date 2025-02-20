import requests
import os
import google.generativeai as genai
import torch
import time

from PIL import Image
from typing import List
from utils import encode_image
from constants.chat_templates import STRICT_MESSAGE, CREATIVE_MESSAGE, TEST_MESSAGE, JAILBREAK_MESSAGE_LLAMA, JAILBREAK_MESSAGE_QWEN
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class Rag:

    def get_answer_from_gemini(self, query, imagePaths):

        print(f"Querying Gemini for query={query}, imagePaths={imagePaths}")

        try:
            genai.configure(api_key=os.environ['GEMINI_API_KEY'])
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            images = [Image.open(path) for path in imagePaths]
            
            chat = model.start_chat()

            response = chat.send_message([*images, query])

            answer = response.text

            print(answer)
            
            return answer
        
        except Exception as e:
            print(f"An error occurred while querying Gemini: {e}")
            return f"Error: {str(e)}"
        

    def get_answer_from_openai(self, query, imagesPaths):
        print(f"Querying OpenAI for query={query}, imagesPaths={imagesPaths}")

        try:    
            payload = self.__get_openai_api_payload(query, imagesPaths)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }
    
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses
    
            answer = response.json()["choices"][0]["message"]["content"]
    
            print(answer)
    
            return answer
    
        except Exception as e:
            print(f"An error occurred while querying OpenAI: {e}")
            return None

    def get_answer_from_llama(query, imagePaths, model, processor, topk=1):        
        try:    
            images = [Image.open(path) for path in imagePaths[:topk]]

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": JAILBREAK_MESSAGE_LLAMA + TEST_MESSAGE}
                    ]
                },
                {
                    "role": "user", "content": [
                        *[{"type": "image"} for _ in images],
                        {"type": "text", "text": query}
                    ]
                }
            ]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=False)

            inputs = processor(
                images,
                input_text,
                add_special_tokens=False, 
                return_tensors="pt"
            ).to(model.device)

            print("Generating output...")

            output = model.generate(**inputs, max_new_tokens=1024, temperature=0.5)
     
            num_input_tokens = inputs["input_ids"].shape[1]
            result = processor.decode(output[0][num_input_tokens:], skip_special_tokens=True)
            print(result)

            return result

        except Exception as e:
            print(f"An error occurred while querying llama: {e}")
            return None
        
    def get_answer_from_qwen(query, imagePaths, model, processor, topk=1):
        """
        Generate response using Qwen2-VL model for multimodal queries
        
        Args:
            query (str): User query text
            imagePaths (list): List of paths to images
            model: Loaded Qwen2VL model instance
            processor: Qwen2VL processor instance
            topk (int): Number of top images to process
        
        Returns:
            str: Generated response text
        """
        print(f"Querying Qwen for query={query}, topk={topk}")
        
        try:
            # Load images
            images = [Image.open(path) for path in imagePaths[:topk]]
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": JAILBREAK_MESSAGE_QWEN + TEST_MESSAGE}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": url} for url in imagePaths[:topk]],
                        {"type": "text", "text": f"{query}"}
                    ]
                }
            ]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process inputs
            inputs = processor(
                text=input_text,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)
            
            print("Generating output...")
            
            # Generate response
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=1.2,
                do_sample=True,
                # top_p=0.9,
                repetition_penalty=1.0
            )
            
            # Decode the response
            num_input_tokens = inputs["input_ids"].shape[1]
            result = processor.decode(output[0][num_input_tokens:], skip_special_tokens=True)
            
            # Clean up the response by removing the input prompt
            if result.startswith(input_text):
                result = result[len(input_text):].strip()
                
            print(f"Generated response: {result}")
            return result
            
        except Exception as e:
            print(f"An error occurred while querying Qwen: {e}")
            return f"Error generating response: {str(e)}"

    # def __get_openai_api_payload(self, query:str, imagesPaths:List[str]):
    #     image_payload = []

    #     for imagePath in imagesPaths:
    #         base64_image = encode_image(imagePath)
    #         image_payload.append({
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": f"data:image/jpeg;base64,{base64_image}"
    #             }
    #         })

    #     payload = {
    #         "model": "gpt-4o",
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": query
    #                     },
    #                     *image_payload
    #                 ]
    #             }
    #         ],
    #         "max_tokens": 1024
    #     }

    #     return payload
    


# if __name__ == "__main__":
#     rag = Rag()
    
#     query = "Based on attached images, how many new cases were reported during second wave peak"
#     imagesPaths = ["covid_slides_page_8.png", "covid_slides_page_8.png"]
    
#     rag.get_answer_from_gemini(query, imagesPaths)