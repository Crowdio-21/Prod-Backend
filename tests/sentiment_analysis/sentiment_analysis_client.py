#!/usr/bin/env python3
"""
Sentiment Analysis Client using Workers

This script demonstrates how to:
1. Connect to the foreman
2. Send text data to workers for sentiment analysis
3. Distribute the work across multiple workers
4. Aggregate and display results
"""

import asyncio
import sys
import os
import json

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from developer_sdk import connect, map as distributed_map, disconnect


def sentiment_worker(text):
    """
    Function to be executed on worker devices for sentiment analysis
    
    Uses a simple lexicon-based approach with positive/negative word lists.
    No regex - pure string operations for maximum compatibility.
    
    Args:
        text: String to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    """
    import time
    import random
    
    start = time.time()
    
    # Simple sentiment lexicon
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'awesome', 'wonderful',
        'fantastic', 'love', 'perfect', 'beautiful', 'best', 'brilliant',
        'nice', 'lovely', 'outstanding', 'superb', 'glad', 'happy', 'pleased',
        'satisfied', 'impressive', 'remarkable', 'successful', 'positive',
        'friendly', 'kind', 'smart', 'clever', 'awesome', 'wonderful'
    }
    
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'poor', 'hate',
        'dislike', 'worst', 'ugly', 'useless', 'broken', 'disappointing',
        'disappointed', 'sad', 'angry', 'upset', 'frustrated', 'annoyed',
        'problematic', 'problem', 'fail', 'failed', 'error', 'difficult',
        'hard', 'complex', 'confusing', 'wrong', 'negative', 'unfriendly'
    }
    
    # Convert to lowercase and split by whitespace, removing punctuation
    text_lower = text.lower()
    # Remove punctuation by replacing with spaces
    for punct in '.,!?;:\'"()[]{}':
        text_lower = text_lower.replace(punct, ' ')
    
    words = text_lower.split()
    
    # Count sentiment words
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    total_sentiment_words = pos_count + neg_count
    
    # Calculate sentiment score
    if total_sentiment_words > 0:
        sentiment = (pos_count - neg_count) / total_sentiment_words
        confidence = min(1.0, total_sentiment_words / (len(words) * 0.3) + random.uniform(0.1, 0.3))
    else:
        sentiment = 0.0
        confidence = random.uniform(0.3, 0.5)
    
    # Calculate subjectivity (rough estimate)
    subjectivity = min(1.0, total_sentiment_words / max(len(words), 1))
    
    # Simulate device processing variability
    time.sleep(random.uniform(0.05, 0.15))
    
    latency_ms = int((time.time() - start) * 1000)
    
    return {
        "text": text[:50] + "..." if len(text) > 50 else text,
        "sentiment": round(sentiment, 3),
        "subjectivity": round(subjectivity, 3),
        "confidence": round(min(1.0, confidence), 3),
        "latency_ms": latency_ms
    }


def split_text_into_sentences(text):
    """
    Split text into sentences for distributed processing
    
    Uses simple string splitting without regex.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Split on sentence delimiters (., !, ?)
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?':
            sentence = current_sentence.strip()
            if sentence:
                sentences.append(sentence)
            current_sentence = ""
    
    # Add any remaining text as a sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences


def aggregate_sentiment_results(results):
    """
    Aggregate sentiment analysis results from multiple workers
    
    Args:
        results: List of sentiment analysis results from workers
        
    Returns:
        Dictionary with aggregated results
    """
    if not results:
        return {
            "overall_sentiment": 0.0,
            "overall_subjectivity": 0.0,
            "avg_confidence": 0.0,
            "sentence_count": 0
        }

    sentiments = [r["sentiment"] for r in results]
    subjectivities = [r["subjectivity"] for r in results]
    confidences = [r["confidence"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    # Calculate weighted average sentiment (higher confidence = higher weight)
    total_weight = sum(confidences)
    if total_weight > 0:
        weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / total_weight
    else:
        weighted_sentiment = sum(sentiments) / len(sentiments)

    return {
        "overall_sentiment": round(weighted_sentiment, 3),
        "overall_subjectivity": round(sum(subjectivities) / len(subjectivities), 3),
        "avg_confidence": round(sum(confidences) / len(confidences), 3),
        "min_sentiment": round(min(sentiments), 3),
        "max_sentiment": round(max(sentiments), 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "sentence_count": len(results)
    }


async def run_distributed_sentiment_analysis(text, foreman_host="localhost", foreman_port=8000):
    """
    Run distributed sentiment analysis using workers
    
    Args:
        text: Text to analyze
        foreman_host: Hostname of the foreman
        foreman_port: Port of the foreman (unused, foreman uses port 9000 for WebSocket)
    """
    print("\n" + "="*70)
    print("🔍 DISTRIBUTED SENTIMENT ANALYSIS")
    print("="*70)
    
    # Connect to foreman
    print(f"\n📡 Connecting to foreman at {foreman_host}:9000...")
    try:
        await connect(foreman_host, 9000)
    except Exception as e:
        print(f"❌ Failed to connect to foreman: {e}")
        return
    
    print("✅ Connected!")

    # Split text into sentences
    print(f"\n📝 Input text: {text[:100]}..." if len(text) > 100 else f"\n📝 Input text: {text}")
    sentences = split_text_into_sentences(text)
    print(f"📊 Split into {len(sentences)} sentences")

    if not sentences:
        print("⚠️  No sentences to analyze")
        await disconnect()
        return

    # Submit distributed tasks
    print(f"\n⏳ Submitting {len(sentences)} sentiment analysis tasks to workers...")
    
    try:
        # Use distributed_map to process sentences across workers
        results = await distributed_map(sentiment_worker, sentences)
        
        print(f"✅ Received {len(results)} results from workers")

        # Aggregate results
        print("\n" + "-"*70)
        print("📊 RESULTS")
        print("-"*70)
        
        aggregated = aggregate_sentiment_results(results)
        
        print(f"\n📈 Overall Sentiment Score: {aggregated['overall_sentiment']:.3f}")
        print(f"   (Range: -1.0 [negative] to +1.0 [positive])")
        
        print(f"\n💭 Overall Subjectivity: {aggregated['overall_subjectivity']:.3f}")
        print(f"   (Range: 0.0 [objective] to 1.0 [subjective])")
        
        print(f"\n🎯 Average Confidence: {aggregated['avg_confidence']:.3f}")
        
        print(f"\n📊 Sentiment Range:")
        print(f"   Min: {aggregated['min_sentiment']:.3f}")
        print(f"   Max: {aggregated['max_sentiment']:.3f}")
        
        print(f"\n⚡ Avg Worker Latency: {aggregated['avg_latency_ms']:.1f}ms")
        
        print(f"\n📋 Sentence Analysis Details:")
        print("-"*70)
        for i, result in enumerate(results, 1):
            sentiment_emoji = "😢" if result["sentiment"] < -0.2 else "😐" if result["sentiment"] < 0.2 else "😊"
            print(f"{i}. {sentiment_emoji} {result['text']}")
            print(f"   Sentiment: {result['sentiment']:.3f} | Subjectivity: {result['subjectivity']:.3f} | Confidence: {result['confidence']:.3f}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"❌ Error during distributed analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await disconnect()


async def main():
    """Main entry point"""
    # Example text for sentiment analysis
    example_text = """
    I absolutely love this product! It's amazing and works perfectly.
    However, the customer service was disappointing.
    The quality is excellent, but the price is too high.
    I would definitely recommend this to my friends.
    Overall, I'm quite satisfied with my purchase.
    """
    
    # Get foreman host from command line args
    foreman_host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    foreman_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    await run_distributed_sentiment_analysis(example_text, foreman_host, foreman_port)


if __name__ == "__main__":
    asyncio.run(main())
