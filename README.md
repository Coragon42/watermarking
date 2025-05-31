# AI Watermarking Research – ERSP 2024  

This repository contains code and resources for our research on watermarking AI-generated content, conducted under the guidance of **Professor Ananth** as part of the **Early Research Scholars Program (ERSP) at UCSB**.  

## 📌 Overview  
This project explores various LLM watermarking techniques to embed and detect watermarks in generated text while preserving semantic integrity. The repository includes:  

- **SynthID** – Experimenting with Google’s SynthID watermarking method.  
- **Soft Watermarking** – Revisiting the pioneer of key-based watermarks.
- **Unigram Watermarking** – Evaluating a similar watermark with a fixed green list for comparison.

## 📂 Repository Contents  
- Collaborative notebooks and Python scripts for different watermarking methods, including custom mass-prompting pipelines for Soft and Unigram watermarks
- Experimental results and research insights
- Spreadsheet benchmarking various attack examples and ideas:
https://docs.google.com/spreadsheets/d/15F7iMyDz2Qb0t_mAk-JszkRt_wM25wpxttvJuKUWmc8/edit?gid=0#gid=0
- Poster presentation

## 👥 Contributors  
- **Zeel Patel**  
- **Brian Sen**
- **Siddhi Mundhra**
- **Emerson Yu** 

## Noted  
This repository primarily consists of collaborative notebooks and code used in our research.  

Environment setup for all three watermarks, after creating and activating environment (python=3.11.11):
- python -m pip install “synthid-text[notebook]” notebook absl-py mock pytest tensorflow-datasets>=4.9.3 SentencePiece accelerate>=0.26.0 safetensors>=0.4.1 bitsandbytes tf-keras
- python -m pip install --upgrade jax jaxlib flax transformers optax
- after installing CUDA Toolkit, install PyTorch accordingly, e.g.:
- python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
- python -m pip install gradio nltk