# 🤖 Physics QA Bot – Motion in a Plane (Class 11)

This is a Transformer-based **Question-Answering (QA) bot** trained specifically on the **"Motion in a Plane"** chapter from Class 11 Physics. The bot is capable of understanding natural language questions related to the chapter and generating contextually relevant answers using a Decoder-Only Transformer architecture.

---

## 📘 Project Overview

This project uses a custom-trained transformer model to generate physics answers based on a trained vocabulary. It has been built using **PyTorch** for the model and **FastAPI** for deploying the API. The frontend is a simple static HTML interface served via FastAPI.

---

## 🧠 Key Features

- ⚙️ **Custom-built Transformer** (Decoder-Only) trained on chapter-wise QA pairs.
- 📚 **Domain-specific**: Focused on "Motion in a Plane" from Class 11 NCERT Physics.
- 🔤 **Custom tokenizer and vocabulary** using special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<SEP>`).
- 🚀 **FastAPI-based REST API** to generate answers to user questions.
- 🖥️ **Interactive frontend** served via `/` endpoint.

---

## 🏗️ Model Architecture

- **Embedding Layer**: Word + Positional Embeddings
- **Transformer Encoder Stack**: 6 layers, 8 attention heads
- **Output Layer**: Linear layer projecting to vocabulary size
- **Input Format**: 
