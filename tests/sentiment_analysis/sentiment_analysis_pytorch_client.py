#!/usr/bin/env python3
"""
PyTorch-based Sentiment Analysis Client using Distributed Workers

This script demonstrates:
1. Distributed NLP inference using PyTorch
2. HuggingFace transformer models (DistilBERT)
3. Task offloading to worker devices
4. Result aggregation and analysis
"""

import asyncio
import sys
import os
import json

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from developer_sdk import connect, map as distributed_map, disconnect


def sentiment_worker_pytorch(text):
    """
    PyTorch-based sentiment analysis worker
    """
    import time
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    start = time.time()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

        print(f"[Worker] Loading model: {MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()  # IMPORTANT for inference

        # Tokenize text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        predicted_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(torch.max(probs).item())

        sentiment = 1.0 if predicted_class == 1 else -1.0

        neg_prob = float(probs[0][0])
        pos_prob = float(probs[0][1])

        latency_ms = int((time.time() - start) * 1000)

        result = {
            "text": text[:50] + "..." if len(text) > 50 else text,
            "sentiment": round(sentiment, 3),
            "confidence": round(confidence, 3),
            "predicted_class": predicted_class,
            "class_name": "positive" if predicted_class == 1 else "negative",
            "neg_probability": round(neg_prob, 3),
            "pos_probability": round(pos_prob, 3),
            "logits": [float(logits[0][0]), float(logits[0][1])],
            "model": MODEL_NAME,
            "tensor_input_shape": list(inputs["input_ids"].shape),
            "tensor_output_shape": list(logits.shape),
            "latency_ms": latency_ms,
            "status": "success"
        }

        print(f"[Worker] Success: {result['class_name'].upper()} | Confidence: {confidence:.3f}")
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()

        latency_ms = int((time.time() - start) * 1000)

        return {
            "text": text[:50] + "..." if len(text) > 50 else text,
            "sentiment": 0.0,
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "status": "error",
            "error": str(e)
        }


# =========================================================
# ✂️ TEXT SPLITTING
# =========================================================
def split_text_into_sentences(text):
    sentences = []
    current = ""

    for ch in text:
        current += ch
        if ch in ".!?":
            sentence = current.strip()
            if sentence:
                sentences.append(sentence)
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences


# =========================================================
#  RESULT AGGREGATION
# =========================================================
def aggregate_sentiment_results(results):
    parsed = []

    for r in results:
        if isinstance(r, dict):
            parsed.append(r)
        elif isinstance(r, str):
            try:
                parsed.append(json.loads(r))
            except:
                try:
                    import ast
                    parsed.append(ast.literal_eval(r))
                except:
                    parsed.append({"status": "error", "error": "Unparseable result"})
        else:
            parsed.append({"status": "error", "error": "Unknown result type"})

    valid = [r for r in parsed if r.get("status") == "success"]

    if not valid:
        return {
            "overall_sentiment": 0.0,
            "avg_confidence": 0.0,
            "sentence_count": 0,
            "error_count": len(results),
            "error": "All results failed or could not be parsed",
            "parsed_results": []
        }

    sentiments = [r["sentiment"] for r in valid]
    confidences = [r["confidence"] for r in valid]
    latencies = [r["latency_ms"] for r in valid]

    total_weight = sum(confidences)
    weighted_sentiment = (
        sum(s * c for s, c in zip(sentiments, confidences)) / total_weight
        if total_weight > 0 else 0.0
    )

    return {
        "overall_sentiment": round(weighted_sentiment, 3),
        "avg_confidence": round(sum(confidences) / len(confidences), 3),
        "min_confidence": round(min(confidences), 3),
        "max_confidence": round(max(confidences), 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "sentence_count": len(valid),
        "error_count": len(results) - len(valid),
        "parsed_results": valid
    }


# =========================================================
#  DISTRIBUTED EXECUTION
# =========================================================
async def run_distributed_sentiment_analysis(text, foreman_host="localhost"):
    print("\n" + "=" * 70)
    print("🔍 DISTRIBUTED PYTORCH SENTIMENT ANALYSIS")
    print("=" * 70)

    print(f"\n📡 Connecting to foreman at {foreman_host}:9000...")
    await connect(foreman_host, 9000)
    print("✅ Connected")

    sentences = split_text_into_sentences(text)
    print(f"📝 Sentences: {len(sentences)}")

    print("\n⏳ Dispatching tasks to workers...")
    results = await distributed_map(sentiment_worker_pytorch, sentences)

    aggregated = aggregate_sentiment_results(results)

    print("\n" + "-" * 70)
    print("📊 RESULTS")
    print("-" * 70)

    print(f"\n📈 Overall Sentiment: {aggregated['overall_sentiment']}")
    print(f"🎯 Avg Confidence: {aggregated['avg_confidence']}")

    if "avg_latency_ms" in aggregated:
        print(f"⚡ Avg Worker Latency: {aggregated['avg_latency_ms']} ms")

    print("\n📋 Sentence-level Results:")
    for i, r in enumerate(aggregated.get("parsed_results", []), 1):
        emoji = "😊" if r["sentiment"] > 0 else "😢"
        print(f"{i}. {emoji} {r['text']}")
        print(f"   Class: {r['class_name']} | Confidence: {r['confidence']}")

    await disconnect()


# =========================================================
#  MAIN
# =========================================================
async def main():
    example_text = """
    I absolutely love this product! It's amazing and works perfectly.
    However, the customer service was disappointing and frustrating.
    The quality is excellent, but the price is too high.
    I would definitely recommend this to my friends.
    Overall, I'm quite satisfied with my purchase.
    """

    foreman_host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    await run_distributed_sentiment_analysis(example_text, foreman_host)


if __name__ == "__main__":
    asyncio.run(main())
