# LSTM Next Word Prediction System

A deep learning-based Next Word Prediction system built using LSTM (Long Short-Term Memory) networks, deployed as a REST API using FastAPI on Railway.

---

## Assignment Details
- **Assignment**: LSTM-Based Sequence Prediction System
- **Task**: Text Prediction (Next Word)
- **Deployment**: FastAPI (REST API)
- **Group Size**: 4 Students

---

## Live API
- **Base URL**: https://lstm-next-word-api-production.up.railway.app
- **Swagger UI**: https://lstm-next-word-api-production.up.railway.app/docs
- **Health Check**: https://lstm-next-word-api-production.up.railway.app/health
- **Predict**: https://lstm-next-word-api-production.up.railway.app/predict

---

## Dataset
- **Name**: WikiText-2
- **Source**: https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/
- **Description**: Clean Wikipedia text, standard NLP benchmark dataset
- **Total Size**: 10.7M characters, 1.79M words
- **Words Used**: 200,000 words
- **Vocabulary Size**: 14,212 unique words

---

## Preprocessing Steps
1. Combined train and validation text
2. Removed WikiText noise (unk, @-@, section headers)
3. Removed punctuation and special characters
4. Converted to lowercase
5. Tokenized using Keras Tokenizer
6. Generated sequences of length 10
7. Split into Train 70% / Validation 15% / Test 15%

---

## LSTM Mathematical Model

### 1. Forget Gate
f(t) = sigmoid(Wf . [h(t-1), x(t)] + bf)
Decides what information to forget from cell state
In our code: recurrent_dropout=0.2 regularizes this gate

### 2. Input Gate
i(t) = sigmoid(Wi . [h(t-1), x(t)] + bi)
g(t) = tanh(Wg . [h(t-1), x(t)] + bg)
Decides what new information to store in memory
In our code: activation=tanh inside LSTM layer

### 3. Cell State (Long Term Memory)
C(t) = f(t) * C(t-1) + i(t) * g(t)
Carries context across entire 10 word sequence
Example: remembers king when processing queen

### 4. Output Gate
o(t) = sigmoid(Wo . [h(t-1), x(t)] + bo)
h(t) = o(t) * tanh(C(t))
Produces hidden state passed to next LSTM layer
In our code: return_sequences=True passes h(t) to Layer 2

### 5. Sequence Learning
Sequence length = 10 words processed one by one
Each word updates cell state and hidden state
Final hidden state passed to Dense layer
Dense(14212) + softmax predicts next word from vocabulary
Example: the king and queen -> model predicts and

---

## Model Architecture

Input (10 words)
      |
Embedding Layer (14212 to 128 dimensions)
      |
LSTM Layer 1 (256 units, return_sequences=True, dropout=0.3)
Learns basic word patterns
      |
LSTM Layer 2 (256 units, return_sequences=False, dropout=0.3)
Learns complex relationships
      |
Dense Layer (256 units, ReLU activation)
      |
Dropout (0.3)
      |
Dense Layer (14212 units, Softmax activation)
      |
Next Word Prediction

Total Parameters: 6,456,964
Trainable Parameters: 6,456,964

---

## Hyperparameter Tuning
Tested 36 combinations across 4 parameters

Parameter        | Values Tested
LSTM Units       | 64, 128, 256
Dropout Rate     | 0.2, 0.3, 0.5
Learning Rate    | 0.001, 0.005
Embedding Dim    | 64, 128

Best Hyperparameters Found:
Parameter        | Best Value
LSTM Units       | 256
Dropout          | 0.3
Learning Rate    | 0.005
Embedding Dim    | 128

---

## Training Results

Metric           | Value
Total Epochs Run | 9
Best Epoch       | 5
Train Loss       | 5.7299
Validation Loss  | 6.6672
Test Loss        | 6.6790
Test Accuracy    | 12.75%
Val vs Test Diff | 0.0118 (No Overfitting)

Callbacks Used:
- ModelCheckpoint: saves best model automatically
- ReduceLROnPlateau: reduces lr when val loss stops improving
- EarlyStopping: stopped at epoch 9, restored best weights from epoch 5

---

## API Endpoints

Method | Endpoint  | Description
GET    | /         | Root - API info
GET    | /health   | Health check - model status
POST   | /predict  | Predict next word

### Sample Request
POST /predict
{
  "text": "the king and queen",
  "top_k": 5
}

### Sample Response
{
  "input_text": "the king and queen",
  "predictions": [
    {"rank": 1, "word": "and",   "probability": "3.43%"},
    {"rank": 2, "word": "the",   "probability": "3.41%"},
    {"rank": 3, "word": "in",    "probability": "2.36%"},
    {"rank": 4, "word": "a",     "probability": "1.14%"},
    {"rank": 5, "word": "to",    "probability": "1.02%"}
  ],
  "model_info": {
    "model_name": "LSTM Next Word Predictor",
    "dataset": "WikiText-2",
    "vocab_size": 14212,
    "sequence_len": 10,
    "test_accuracy": "12.75%",
    "test_loss": "6.6790"
  }
}

---

## Repository Structure

lstm-next-word-api/
    main.py          - FastAPI application code
    requirements.txt - Python dependencies
    tokenizer.pkl    - Saved Keras tokenizer
    runtime.txt      - Python version for Railway
    README.md        - Project documentation

Model stored on Google Drive (too large for GitHub):
https://drive.google.com/file/d/1iYUgy28_cQrWgPxsOevhjuOJtysRiw4i/view

---

## Dependencies
fastapi
uvicorn
numpy
tensorflow
pydantic
python-multipart
gdown

---

## Deployment
Platform: Railway (https://railway.app)
Python Version: 3.11.9
Model auto-downloaded from Google Drive on startup
Public URL available 24/7

---

## Why Test Accuracy is 12.75%

Vocabulary has 14,212 words
Random guessing accuracy = 0.007%
Our LSTM accuracy = 12.75%
Our model is 1800x better than random guessing

Real world comparison:
Random Guess  | 0.007%
Our LSTM      | 12.75%
GPT-4         | 60-70%

GPT-4 was trained on trillions of words for months
Our model was trained on 200,000 words in 9 minutes
Result is excellent for assignment level model



---

## References
- FastAPI Documentation: https://fastapi.tiangolo.com/
- TensorFlow/Keras Documentation: https://www.tensorflow.org/api_docs
- WikiText-2 Dataset: https://pytorch.org/text/stable/datasets.html
- Railway Deployment: https://railway.app
- Keras LSTM: https://keras.io/examples/nlp/text_generation/
