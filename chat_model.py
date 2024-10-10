import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Class to handle the T5 model for generating responses
@st.cache_resource
class ChatModel:
    def __init__(self):
        # Load the tokenizer and model from the pre-trained directory
        model_dir = os.getenv("OUTPUT_MODEL_DIR")
        print(f"Loading model from: {model_dir}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)

        # Ensure the padding token is set correctly
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')

    # Function to generate a response
    def generate_response(self, prompt, history=None, max_length=200, num_return_sequences=1, temperature=0.9, top_k=50, top_p=0.9):
        # If there is a message history, concatenate it
        if history:
            context = " ".join([f"{msg['role']}: {msg['content']}" for msg in history])
            input_prompt = f"{context} {prompt}"
        else:
            input_prompt = prompt

        # Tokenize the input prompt with appropriate padding
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt', padding='longest').to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate text
        outputs = self.model.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=num_return_sequences, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p,
            do_sample=True
        )
        
        # Decode the generated text, ensuring special tokens are skipped
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for output in outputs]
        cleaned_texts = [self._post_process(text) for text in generated_texts]  # Post-process the generated texts

        return cleaned_texts

    def _post_process(self, text):
        # Remove unwanted tokens and perform basic corrections
        text = text.replace("<pad>", "").strip()
        # Implement further corrections if necessary (e.g., grammar correction)
        return text
