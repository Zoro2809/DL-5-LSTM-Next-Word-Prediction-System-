# ============================================
# LSTM Next Word Prediction - FastAPI App
# ============================================

import numpy as np
import pickle
import os
import gdown
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from tensorflow.keras.models import load_model

# ============================================
# Download Model from Google Drive
# ============================================
MODEL_PATH     = 'next_word_model.keras'
TOKENIZER_PATH = 'tokenizer.pkl'
SEQ_LENGTH     = 10

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        'https://drive.google.com/uc?id=1iYUgy28_cQrWgPxsOevhjuOJtysRiw4i',
        MODEL_PATH,
        quiet=False
    )
    print(" Model downloaded!")

# Load model
print("Loading model...")
model = load_model(MODEL_PATH)

# Load tokenizer
print("Loading tokenizer...")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

VOCAB_SIZE = len(tokenizer.word_index) + 1
print(f" Ready! Vocab size: {VOCAB_SIZE}")

# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="LSTM Next Word Prediction API",
    description="""
## LSTM Based Next Word Prediction System

Built as part of AI Systems Assignment.

- **Dataset**      : WikiText-2 (Wikipedia text)
- **Model**        : Stacked LSTM (2 layers, 256 units)
- **Vocab Size**   : 14,212 words
- **Test Accuracy**: 12.75%
- **Test Loss**    : 6.6790

### How to use:
1. Click on **/predict** endpoint below
2. Click **Try it out**
3. Enter your text in the box
4. Click **Execute**
5. See the top predicted next words!
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================
# Request and Response Models
# ============================================
class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text" : "the king and queen",
                "top_k": 5
            }
        }
    )
    text  : str
    top_k : int = 5

class PredictResponse(BaseModel):
    input_text  : str
    predictions : list
    model_info  : dict

# ============================================
# Endpoints
# ============================================
@app.get("/")
def root():
    return {
        "message"    : "LSTM Next Word Prediction API is Live!",
        "description": "Send POST request to /predict with your text",
        "endpoints"  : {
            "root"   : "GET  /",
            "health" : "GET  /health",
            "predict": "POST /predict",
            "docs"   : "GET  /docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status"       : "healthy",
        "model"        : "LSTM - WikiText2",
        "vocab_size"   : VOCAB_SIZE,
        "sequence_len" : SEQ_LENGTH,
        "test_accuracy": "12.75%",
        "test_loss"    : "6.6790"
    }

@app.post("/predict", response_model=PredictResponse)
def predict_next_word(request: PredictRequest):

    # Validation
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty"
        )

    if request.top_k < 1 or request.top_k > 10:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 10"
        )

    # Tokenize input
    token_list = tokenizer.texts_to_sequences([request.text.lower()])[0]

    # Pad or trim to sequence length
    if len(token_list) < SEQ_LENGTH:
        token_list = [0] * (SEQ_LENGTH - len(token_list)) + token_list
    else:
        token_list = token_list[-SEQ_LENGTH:]

    token_array = np.array(token_list).reshape(1, SEQ_LENGTH)

    # Predict
    preds         = model.predict(token_array, verbose=0)[0]
    top_k_indices = preds.argsort()[-request.top_k:][::-1]

    predictions = []
    for rank, idx in enumerate(top_k_indices, 1):
        word = tokenizer.index_word.get(idx, 'unknown')
        prob = round(float(preds[idx]) * 100, 2)
        predictions.append({
            "rank"       : rank,
            "word"       : word,
            "probability": f"{prob}%"
        })

    return PredictResponse(
        input_text  = request.text,
        predictions = predictions,
        model_info  = {
            "model_name"   : "LSTM Next Word Predictor",
            "dataset"      : "WikiText-2",
            "vocab_size"   : VOCAB_SIZE,
            "sequence_len" : SEQ_LENGTH,
            "test_accuracy": "12.75%",
            "test_loss"    : "6.6790"
        }
    )

# ============================================
# Run
# ============================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)