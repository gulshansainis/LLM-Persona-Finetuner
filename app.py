# app.py (Definitive Final Version - Polished UI with Infographic-Inspired CSS)
# Author: Gulshan Saini
# Date: August 29, 2025
# Description: A beautifully styled, hardware-aware Streamlit app that compares 
# a base LLM with a fine-tuned version, featuring a custom, professional UI.

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

# --- Configuration ---
class AppConfig:
    BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    FINETUNED_MODEL_PATH = "./Qwen3-4B-Financial-Analyst"

# --- Model Loading ---
@st.cache_resource
def load_models():
    # (The model loading function is perfect and remains unchanged)
    if torch.cuda.is_available():
        device = "cuda"
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
    elif torch.backends.mps.is_available():
        device = "mps"
        model_kwargs = {"dtype": torch.bfloat16}
    else:
        device = "cpu"
        model_kwargs = {"dtype": torch.bfloat16}
    
    tokenizer = AutoTokenizer.from_pretrained(AppConfig.BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(AppConfig.BASE_MODEL_ID, **model_kwargs)
    
    if "device_map" not in model_kwargs:
        base_model.to(device)

    ft_model = PeftModel.from_pretrained(base_model, AppConfig.FINETUNED_MODEL_PATH)
    return tokenizer, base_model, ft_model, device

def generate_response(tokenizer, model, chat_prompt, device, is_finetuned=False):
    # (The generation function is perfect and remains unchanged)
    if is_finetuned:
        prompt_text = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = chat_prompt[-1]['content']

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id, temperature=0.7)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# --- Custom CSS for a Professional, Infographic-Inspired Look ---
def load_css():
    st.markdown("""
    <style>
        /* Import a clean, modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* General App Styling */
        .stApp {
            background-color: #F8F9FE; /* Light blue/purple background from infographic */
            font-family: 'Inter', sans-serif;
        }

        /* Main Title and Headers */
        h1 {
            color: #1A202C;
            font-weight: 700;
        }
        h3 { /* For "Enter a Financial Question" */
            color: #2D3748;
            font-weight: 600;
            padding-top: 1rem;
        }
        
        /* Main content area padding */
        .st-emotion-cache-z5fcl4 {
             padding-top: 2rem;
        }

        /* Input Box & Button Styling */
        .stTextInput>div>div>input {
            background-color: #FFFFFF;
            border: 1px solid #CBD5E0;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #5A67D8; /* Vibrant blue/purple from infographic */
            color: white;
            border: none;
            font-weight: 600;
            padding: 10px 24px;
            transition: background-color 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #434190;
            color: white;
        }
        
        /* --- The Main Attraction: Prominent Comparison Cards --- */
        .response-card {
            background-color: #FFFFFF;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 20px -8px rgba(0, 0, 0, 0.1);
            border: 1px solid #E2E8F0;
            height: 480px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        .card-header {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #F1F5F9;
        }
        .card-content {
            font-size: 1rem;
            line-height: 1.7;
            color: #4A5568;
        }
        .card-header.base {
            color: #5A67D8; /* Vibrant blue/purple */
        }
        .card-header.finetuned {
            color: #38B2AC; /* Teal from infographic */
        }
        
        /* Hide Streamlit's default footer */
        footer {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

# Helper function to format text for safe HTML rendering
def format_for_html(text):
    # Basic bolding for "**text**"
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Convert newlines to <br> tags
    return text.replace('\n', '<br>')

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="LLM Fine-Tuning: Before vs. After", layout="wide")
    load_css()

    st.title("⚖️ LLM Fine-Tuning: Before vs. After")
    st.markdown("##### Comparing a base Qwen3-4B model with a version fine-tuned to be a Financial Analyst.")
    st.markdown("---")

    tokenizer, base_model, ft_model, device = load_models()
    
    st.subheader("Enter a Financial Question:")
    user_question = st.text_input("Prompt", label_visibility="collapsed", placeholder="Try 'Should I buy Apple or Google stock?'")

    if st.button("Generate Comparison"):
        if not user_question:
            st.warning("Please enter a question.")
        else:
            system_message = "You are a financial analyst. Your task is to reframe a user's direct question into a neutral, data-driven analysis of the pros and cons for each option. Do not give advice or recommendations."
            chat = [{"role": "system", "content": system_message}, {"role": "user", "content": user_question}]
            
            with st.spinner(f"Generating responses on {device.upper()}..."):
                base_response = generate_response(tokenizer, base_model, chat, device, is_finetuned=False)
                ft_response = generate_response(tokenizer, ft_model, chat, device, is_finetuned=True)
            
            st.subheader("Comparison of Responses")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="response-card">
                    <div class="card-header base">🤖 Base Model (Qwen3-4B)</div>
                    <div class="card-content">{format_for_html(base_response)}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="response-card">
                    <div class="card-header finetuned">👨‍💼 Fine-Tuned Financial Analyst</div>
                    <div class="card-content">{format_for_html(ft_response)}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()