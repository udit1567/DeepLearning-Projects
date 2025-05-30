# PhysicsQA-GPT: Transformer-Based Q&A Model for "Motion in a Plane" (Class 11 Physics)

PhysicsQA-GPT is a domain-specific Question-Answering (QA) bot focused exclusively on the "Motion in a Plane" chapter from Class 11 Physics. Built using a custom Decoder-Only Transformer architecture, the system provides accurate, context-aware answers to user questions, leveraging a tailored vocabulary and tokenizer. The project is implemented in PyTorch and deployed via FastAPI, with an interactive web frontend.

---

## 📘 Project Overview

PhysicsQA-GPT is designed to help students master the concepts of "Motion in a Plane," including vectors, projectile motion, and two-dimensional kinematics. It leverages a deep learning model trained on curated QA pairs, offering fast and relevant responses to natural language queries.

### Key Learning Areas Covered

- Vectors and their resolution in a plane  
- Scalar and vector quantities  
- Equations of motion in two dimensions  
- Projectile motion and uniform circular motion  
- Application of Newton’s laws in a plane  

---

## 🧠 Key Features

- **Custom Decoder-Only Transformer**: 6-layer, 8-head attention model trained on chapter-specific QA data.  
- **Domain-Specific Knowledge**: Focused solely on "Motion in a Plane" (Class 11 Physics).  
- **Custom Tokenizer & Vocabulary**: Utilizes special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<SEP>`) for efficient sequence processing.  
- **REST API with FastAPI**: Exposes endpoints for answer generation and serves a static HTML frontend.  
- **Interactive Frontend**: Simple web interface for user interaction.  

---

## 🏗️ Model Architecture

| Component         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Embedding Layer   | Word and positional embeddings for input tokens                            |
| Transformer Encoder | 6 stacked layers, each with 8 attention heads, masked self-attention, and feed-forward nets |
| Output Layer       | Linear projection to vocabulary size                                       |
| Special Tokens     | `<PAD>`, `<BOS>`, `<EOS>`, `<SEP>` for sequence control                    |

### Input Format

- **Input Sequence**: `<BOS> question tokens <SEP> answer tokens <EOS>`  
- **Tokenization**: Lowercased, punctuation separated, split on whitespace.  

---

## 🖥️ API & Frontend

### FastAPI Endpoints

- `GET /` — Serves the static HTML frontend.  
- `POST /generate-answer` — Accepts a JSON payload with a `question` field, returns the generated answer.

### Frontend

- Simple HTML interface for entering questions and displaying answers, served from the `/static` directory.

---

## 🚀 Usage

### 1. Clone and Install Dependencies

```bash
git clone <repo_url>
cd physicsqa-gpt
pip install -r requirements.txt

![Screenshot 2025-05-30 103538](https://github.com/user-attachments/assets/da042d83-2a1f-49db-97c5-4f0d19205844)

![Screenshot 2025-05-30 103448](https://github.com/user-attachments/assets/6763ab6b-c4d2-4f44-89b8-7e5095ddaf6d)


