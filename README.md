# emotions-recognition

Multi-label emotion classification using a fine-tuned RoBERTa model on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset (28 emotion labels).

## Setup

```bash
pip install -r requirements.txt
```

## Training The Model

Training is done in the notebook `notebooks/train_model.ipynb`.

The notebook expects:
- Dataset at `datasets/go_emotions_dataset.csv`
- Local RoBERTa files in `models/`

### Base RoBERTa Weights (Downloaded Manually)

The base RoBERTa model was downloaded manually from Hugging Face and stored in `models/`.
This directory contains the model/tokenizer files used by the training notebook (for example: `config.json`, tokenizer files, and `roberta-base.safetensors`).

By default, the notebook uses:
- `ROBERTA_MODEL_PATH=../models/roberta-base.safetensors`
- `ROBERTA_TOKENIZER_PATH` resolved from the same `models/` directory

### Run Training

1. Open `notebooks/train_model.ipynb`.
2. Run all cells from top to bottom.
3. The notebook will:
  - Load and split the GoEmotions dataset
  - Tokenize text with RoBERTa
  - Fine-tune a multi-label classifier
  - Save the best checkpoint based on validation micro-F1

### Trained Model Location

The best trained checkpoint is saved to:

`artifacts/roberta-goemotions/best_model.pt`

## CLI Usage

```bash
# Classify one or more texts
python main.py "I'm so happy for you!" "This is terrible."

# Interactive mode
python main.py

# Custom threshold
python main.py -t 0.3 "I can't believe this happened"
```

## API Usage

Start the server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Classify texts into emotions |
| GET | `/health` | Health check & list of labels |

### POST `/predict`

**Request body:**

```json
{
  "texts": ["I'm so happy for you!", "This is really frustrating."],
  "threshold": 0.5
}
```

- `texts` (required): list of strings to classify (max 64).
- `threshold` (optional, default `0.5`): minimum probability to include an emotion.

**Response:**

```json
{
  "predictions": [
    {"joy": 0.92, "love": 0.68},
    {"annoyance": 0.85, "anger": 0.52}
  ]
}
```

### Example with `curl`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Thank you so much!"]}'
```

### Example with Python `requests`

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"texts": ["I feel great today!"]},
)
print(response.json())
```

## Labels

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
