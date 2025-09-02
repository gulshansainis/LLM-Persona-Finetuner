# ⚖️ LLM Persona Finetuner

A project that demonstrates how to fine-tune a powerful, open-source Large Language Model (`Qwen3-4B`) to adopt a specific, expert persona. This repository showcases the entire professional workflow: from creating a high-quality custom dataset, to performing memory-efficient fine-tuning with QLoRA, and finally building an interactive application to prove the success of the transformation.

---

### 🚀 Live Demo

The application provides a side-by-side comparison of the base model's generic response versus the fine-tuned model's expert, persona-driven answer. Notice how the fine-tuned model correctly reframes direct questions into a neutral, data-driven analysis, a skill it learned through fine-tuning.

![Demo GIF showing the comparison](demo.gif)

---

### 💡 The Core Concept: Changing Internal Behavior

This project explores a key area of LLM customization. While my [RAG Chatbot Pro](https://github.com/gulshansainis/RAG-Chatbot-Pro) project focused on making an LLM knowledgeable about *external* data, this project focuses on changing the *internal* behavior and communication style of the model itself.

*   **The Problem:** Powerful base models like `Qwen3-4B-Instruct` are excellent at following general instructions, but they lack a specialized persona and can fail in unpredictable ways (e.g., giving direct financial advice, getting stuck in repetitive loops).
*   **The Solution:** By fine-tuning the model on a small, high-quality, and hyper-focused dataset, we can teach it a new, abstract skill. In this case, we transform a generic chatbot into a robust Financial Analyst that reframes questions into neutral pros-and-cons analyses.

---

### 🛠️ Tech Stack

*   **Training Environment:** Google Colab (leveraging a free NVIDIA T4 GPU)
*   **Core Libraries:** Hugging Face `transformers`, `peft`, and `trl`
*   **Fine-Tuning Technique:** QLoRA (Quantized Low-Rank Adaptation)
*   **Base Model:** [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
*   **UI Framework:** Streamlit
*   **Language:** Python

---

### 📦 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gulshansainis/LLM-Persona-Finetuner.git
    cd LLM-Persona-Finetuner
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **For NVIDIA GPU users:** To enable 4-bit quantization for the fastest performance, also install `bitsandbytes`:
    ```bash
    pip install bitsandbytes
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

*(Note: The fine-tuned model adapter is included in this repository for direct use. The training script (`finetune.py`) and dataset are provided for full reproducibility.)*