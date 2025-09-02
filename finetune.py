# finetune.py
# Author: Gulshan Saini
# Date: August 29, 2025
# Description: This script fine-tunes a Qwen3-4B-Instruct model to adopt the persona
# of a financial analyst. It uses QLoRA for memory-efficient training and includes
# a validation function to test the model immediately after training.
# Designed to be run in a Google Colab environment with a GPU.

# --- 1. Installation ---
# Install the necessary libraries from Hugging Face and PyTorch ecosystem.
!pip install -U transformers torch peft accelerate bitsandbytes trl datasets

# --- 2. Imports ---
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import os

# --- 3. Configuration ---
# A single class to manage all hyperparameters and paths.
class FinetuneConfig:
    # Model and tokenizer identifiers from Hugging Face Hub
    MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Path to the local dataset file
    DATASET_PATH = "dataset.jsonl"
    
    # The name of the folder where the trained adapter will be saved
    NEW_MODEL_NAME = "Qwen3-4B-Financial-Analyst"

# --- 4. Main Training Logic ---
def train_model():
    """
    Handles the entire fine-tuning process, from loading data to saving the model.
    """
    config = FinetuneConfig()

    print("--- Starting Fine-Tuning Process ---")
    print(f"Step 1: Loading dataset from {config.DATASET_PATH}")
    dataset = load_dataset("json", data_files=config.DATASET_PATH, split="train")

    print(f"Step 2: Configuring model loading for {config.MODEL_ID}")
    # Configure 4-bit quantization (QLoRA) to reduce memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Configure PEFT (LoRA) to train only a small number of parameters
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    print("Step 3: Configuring the trainer with SFTConfig")
    training_args = SFTConfig(
        output_dir=config.NEW_MODEL_NAME,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        fp16=True,
        max_length=512,
        model_init_kwargs={"quantization_config": bnb_config, "device_map": "auto"},
        report_to="none",
    )

    print("Step 4: Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=config.MODEL_ID,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Step 5: Beginning model training")
    trainer.train()

    print("Step 6: Saving the trained LoRA adapter")
    trainer.save_model()
    
    print(f"\n--- Fine-tuning complete! Adapter saved to ./{config.NEW_MODEL_NAME} ---")

# --- 5. Validation Logic ---
def validate_model():
    """
    Loads the newly trained model adapter and runs it against a series of test questions.
    """
    config = FinetuneConfig()
    print("\n" + "="*50)
    print("      STARTING VALIDATION TEST")
    print("="*50)

    # Load the base model and tokenizer
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    # Load the PEFT adapter
    ft_model = PeftModel.from_pretrained(base_model, f"./{config.NEW_MODEL_NAME}")

    validation_questions = [
        "Which is a better investment, Tesla or Ford?",
        "Are government bonds a safe place to put my money right now?",
        "What is your outlook on the housing market for the next year?",
        "Should I invest in the latest AI startup stocks? They seem to be going up a lot.",
        "Is it better to pay off my student loans or invest in the stock market?"
    ]
    
    system_message = "You are a financial analyst. Your task is to reframe a user's direct question into a neutral, data-driven analysis of the pros and cons for each option. Do not give advice or recommendations."
    
    for question in validation_questions:
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = ft_model.generate(**inputs, max_new_tokens=250, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print("\n" + "="*50)
        print(f"❓ QUESTION: {question}")
        print("="*50)
        print(response.strip())
        print("-" * 50)

    print("\n--- Validation Test Complete ---")

# --- 6. Execution ---
if __name__ == "__main__":
    train_model()
    validate_model()