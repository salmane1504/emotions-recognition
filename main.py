import argparse
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "roberta-goemotions", "best_model.pt")
MAX_LENGTH = 128
DEFAULT_THRESHOLD = 0.5


def load_model(device: torch.device):
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    config.num_labels = len(LABELS)
    config.problem_type = "multi_label_classification"
    config.id2label = {i: lbl for i, lbl in enumerate(LABELS)}
    config.label2id = {lbl: i for i, lbl in enumerate(LABELS)}
    if device.type == "mps":
        config.attention_probs_dropout_prob = 0.0

    model = AutoModelForSequenceClassification.from_config(config)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def predict(model, tokenizer, texts: list[str], device: torch.device, threshold: float):
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad(), torch.amp.autocast(device.type):
        logits = model(**encodings).logits

    probs = torch.sigmoid(logits.float()).cpu().numpy()
    results = []
    for row in probs:
        emotions = {LABELS[i]: float(row[i]) for i in range(len(LABELS)) if row[i] >= threshold}
        if not emotions:
            best_idx = row.argmax()
            emotions = {LABELS[best_idx]: float(row[best_idx])}
        results.append(dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)))
    return results


def main():
    parser = argparse.ArgumentParser(description="GoEmotions inference with fine-tuned RoBERTa")
    parser.add_argument("texts", nargs="*", help="Text(s) to classify. Omit for interactive mode.")
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Probability threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: mps, cuda, or cpu (auto-detected if omitted)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading model on {device}...")
    model = load_model(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    print("Model ready.\n")

    if args.texts:
        results = predict(model, tokenizer, args.texts, device, args.threshold)
        for text, emotions in zip(args.texts, results):
            print(f"Text: {text}")
            for emotion, prob in emotions.items():
                print(f"  {emotion}: {prob:.4f}")
            print()
    else:
        print("Interactive mode — type a sentence and press Enter (Ctrl+C to quit):\n")
        try:
            while True:
                text = input(">>> ").strip()
                if not text:
                    continue
                results = predict(model, tokenizer, [text], device, args.threshold)
                for emotion, prob in results[0].items():
                    print(f"  {emotion}: {prob:.4f}")
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")


if __name__ == "__main__":
    main()
