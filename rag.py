import google.generativeai as genai

from PIL import Image
from constants.chat_templates import TEST_MESSAGE, JAILBREAK_MESSAGE_LLAMA, JAILBREAK_MESSAGE_QWEN

class Rag:
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

            output = model.generate(**inputs, max_new_tokens=1024, temperature=0.5)
     
            num_input_tokens = inputs["input_ids"].shape[1]
            result = processor.decode(output[0][num_input_tokens:], skip_special_tokens=True)

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
            
            return result
            
        except Exception as e:
            print(f"An error occurred while querying Qwen: {e}")
            return f"Error generating response: {str(e)}"