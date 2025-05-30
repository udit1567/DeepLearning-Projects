from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
import torch
import torch.nn as nn
import json

#loading vocab
with open("vocab.json", "r") as f:
    vocab = json.load(f)
    token_to_id = vocab["token_to_id"]

id_to_token = {i: token for token, i in token_to_id.items()}

def tokenize(text):
    return text.lower().replace("?", " ?").replace(",", " ,").replace(".", " .").split()


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = x.transpose(0, 1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask)
        x = x.transpose(0, 1)
        return self.fc_out(x)
def generate_answer(model, question, max_len=64):
    model.eval()
    with torch.no_grad():
        # Encode question part
        q_tokens = ['<BOS>'] + tokenize(question) + ['<SEP>']
        input_ids = [token_to_id[t] for t in q_tokens]
        start_len = len(input_ids)

        # Generate tokens after <SEP>
        for _ in range(max_len - len(input_ids)):
            x = torch.tensor([input_ids], device=device)
            logits = model(x)
            next_token_id = logits[0, -1].argmax().item()
            if id_to_token[next_token_id] == '<EOS>':
                break
            input_ids.append(next_token_id)

        # Only return generated answer tokens (after <SEP>)
        answer_ids = input_ids[start_len:]
        return " ".join([
            id_to_token[t] for t in answer_ids if id_to_token[t] not in ["<EOS>", "<PAD>"]
        ])
    
class Question(BaseModel):
    question: str

app = FastAPI()

# Load model and tokenizer vocab on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model with same hyperparams as trained
vocab_size = len(token_to_id)
model = DecoderOnlyTransformer(vocab_size).to(device)
model.load_state_dict(torch.load("./model/decoder_only_transformer_model.pth", map_location=device))
model.eval()

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def serve_html():
    return FileResponse("static/index.html")


#Endpoint for generating the output
@app.post("/generate-answer")
def generate_answer_api(item: Question):
    question = item.question
    try:
        answer = generate_answer(model, question)
        return {"answer": answer}
    except KeyError as e:
        # In case of unknown tokens
        raise HTTPException(status_code=400, detail=f"Unknown token in input: {e}")

