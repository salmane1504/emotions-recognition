# emotions-recognition

Multi-label emotion classification using a fine-tuned RoBERTa model on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset (28 emotion labels).

## Setup

```bash
pip install -r requirements.txt
```

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
