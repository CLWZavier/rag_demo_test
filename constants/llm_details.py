MODELS = {
    "meta-llama/Llama-3.2-11B-Vision-Instruct": """
        The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. The models outperform many of the available open source and closed multimodal models on common industry benchmarks.

        Model Developer: Meta

        Model Architecture: Llama 3.2-Vision is built on top of Llama 3.1 text-only model, which is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. To support image recognition tasks, the Llama 3.2-Vision model uses a separately trained vision adapter that integrates with the pre-trained Llama 3.1 language model. The adapter consists of a series of cross-attention layers that feed image encoder representations into the core LLM.

        Supported Languages: For text only tasks, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported. Llama 3.2 has been trained on a broader collection of languages than these 8 supported languages. Note for image+text applications, English is the only language supported.

        Developers may fine-tune Llama 3.2 models for languages beyond these supported languages, provided they comply with the Llama 3.2 Community License and the Acceptable Use Policy. Developers are always expected to ensure that their deployments, including those that involve additional languages, are completed safely and responsibly.

        Llama 3.2 Model Family: Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability.

        Model Release Date: Sept 25, 2024

        Status: This is a static model trained on an offline dataset. Future versions may be released that improve model capabilities and safety.

        License: Use of Llama 3.2 is governed by the Llama 3.2 Community License (a custom, commercial license agreement).

        Feedback: Where to send questions or comments about the model Instructions on how to provide feedback or comments on the model can be found in the model README. For more technical information about generation parameters and recipes for how to use Llama 3.2-Vision in applications, please go here.

        Intended Use Cases: Llama 3.2-Vision is intended for commercial and research use. Instruction tuned models are intended for visual recognition, image reasoning, captioning, and assistant-like chat with images, whereas pretrained models can be adapted for a variety of image reasoning tasks. Additionally, because of Llama 3.2-Vision’s ability to take images and text as inputs, additional use cases could include:

        1. Visual Question Answering (VQA) and Visual Reasoning: Imagine a machine that looks at a picture and understands your questions about it.
        2. Document Visual Question Answering (DocVQA): Imagine a computer understanding both the text and layout of a document, like a map or contract, and then answering questions about it directly from the image.
        3. Image Captioning: Image captioning bridges the gap between vision and language, extracting details, understanding the scene, and then crafting a sentence or two that tells the story.
        4. Image-Text Retrieval: Image-text retrieval is like a matchmaker for images and their descriptions. Similar to a search engine but one that understands both pictures and words.
        5. Visual Grounding: Visual grounding is like connecting the dots between what we see and say. It’s about understanding how language references specific parts of an image, allowing AI models to pinpoint objects or regions based on natural language descriptions.
    """,
    "Qwen/Qwen2-VL-7B-Instruct": """
        What’s New in Qwen2-VL?
        Key Enhancements:
        - SoTA understanding of images of various resolution & ratio: Qwen2-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.

        - Understanding videos of 20min+: Qwen2-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.

        - Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.

        - Multilingual Support: to serve global users, besides English and Chinese, Qwen2-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.
    """,	
}
