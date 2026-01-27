#!/usr/bin/env python3
"""
Distributed DAG-Based Sentiment Analysis Client

This script demonstrates using workers to perform DAG-based sentiment analysis
across distributed PC and Android workers for enhanced precision.

The DAG approach provides:
- Multi-level sentiment analysis (word, phrase, sentence, document)
- Proper handling of negations and intensifiers
- More precise and interpretable results
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from developer_sdk import connect, map as distributed_map, disconnect
from tests.DAG.dag_sentiment_analyzer import analyze_text_with_dag


def dag_sentiment_analysis_worker(message_batch):
    """
    Worker function that performs DAG-based sentiment analysis on a batch of messages.
    
    Args:
        message_batch: List of dicts, each with 'text', 'message_id', and optional 'use_ml'
    
    Returns:
        List of dicts with comprehensive sentiment analysis for each message
    """
    import time
    import os
    from tests.DAG.dag_sentiment_analyzer import SentimentDAG
    
    worker_id = os.getpid()
    
    # Handle both single message (dict) and batch (list) formats
    if isinstance(message_batch, dict):
        message_batch = [message_batch]
    
    print(f"[Worker {worker_id}] Processing batch of {len(message_batch)} messages")
    
    results = []
    
    for message_data in message_batch:
        start = time.time()
        
        text = message_data.get('text', '')
        message_id = message_data.get('message_id', 0)
        use_ml = message_data.get('use_ml', False)
        
        print(f"[Worker {worker_id}] Processing message {message_id}")
        
        # Handle empty text
        if not text or not text.strip():
            results.append({
                "message_id": message_id,
                "text_preview": "(empty message)",
                "sentiment_score": 0.5,
                "sentiment_label": "NEUTRAL",
                "confidence": 0.0,
                "advice": ["Empty message - unable to analyze"],
                "method": "DAG",
                "status": "success",
                "latency_ms": 0,
                "worker_id": worker_id
            })
            continue
        
        try:
            # Build and analyze with DAG
            dag = SentimentDAG()
            dag.build_from_text(text, message_id)
            dag.compute_sentiments(use_ml=use_ml)
            result = dag.get_final_sentiment()
            
            # Add DAG visualization
            dag_visualization = dag.visualize_dag()
            result["dag_visualization"] = dag_visualization
            
            # Generate advice based on sentiment
            sentiment_score = result["sentiment_score"]
            
            if sentiment_score < 0.3:
                advice = [
                    "🔴 CRITICAL: Customer is highly dissatisfied",
                    "Acknowledge frustration immediately",
                    "Show empathy and take ownership",
                    "Escalate if needed"
                ]
            elif sentiment_score < 0.5:
                advice = [
                    "🟡 CAUTION: Customer showing dissatisfaction",
                    "Be proactive and validate concerns",
                    "Show clear action steps"
                ]
            elif sentiment_score < 0.7:
                advice = [
                    "🟢 NEUTRAL: Customer is engaged",
                    "Build rapport and set clear expectations",
                    "Maintain professional tone"
                ]
            else:
                advice = [
                    "💚 EXCELLENT: Customer is satisfied",
                    "Maintain positive momentum",
                    "Show appreciation"
                ]
            
            result["advice"] = advice
            result["latency_ms"] = int((time.time() - start) * 1000)
            result["method"] = "DAG (Distributed)"
            result["status"] = "success"
            result["worker_id"] = worker_id
            
            results.append(result)
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"[DAG Worker Error] {error_msg}")
            traceback.print_exc()
            
            results.append({
                "message_id": message_id,
                "text_preview": text[:100] if text else "",
                "sentiment_score": 0.5,
                "sentiment_label": "ERROR",
                "confidence": 0.0,
                "advice": [f"Analysis failed: {error_msg}"],
                "latency_ms": int((time.time() - start) * 1000),
                "method": "DAG (Distributed)",
                "status": "failed",
                "error": error_msg,
                "worker_id": worker_id
            })
    
    print(f"[Worker {worker_id}] Completed batch of {len(results)} messages")
    return results


async def load_dataset(sample_size=None):
    """Load customer support dataset"""
    import kagglehub
    import pandas as pd
    
    CACHE_FILE = Path("dataset_cache.json")
    
    # Check cache first
    if CACHE_FILE.exists():
        print("📦 Loading dataset from cache...")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} messages from cache")
        return data
    
    print("📥 Downloading customer support dataset from Kaggle...")
    print("   (This will be cached for future runs)")
    
    try:
        # Download dataset
        path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Load CSV
        csv_path = Path(path) / "twcs.csv"
        if not csv_path.exists():
            csv_path = Path(path) / "twcs" / "twcs.csv"
        
        df = pd.read_csv(csv_path)
        print(f"📊 Total rows in dataset: {len(df)}")
        
        # Filter to customer messages (inbound)
        customer_messages = df[df['inbound'] == True].copy()
        print(f"👥 Customer messages: {len(customer_messages)}")
        
        # Sample if requested
        if sample_size and sample_size < len(customer_messages):
            customer_messages = customer_messages.sample(n=sample_size, random_state=42)
            print(f"🎲 Sampled {sample_size} messages")
        
        # Convert to list of dicts
        data = []
        for idx, row in customer_messages.iterrows():
            text = str(row['text']) if pd.notna(row['text']) else ""
            if text and len(text.strip()) > 10:  # Filter out very short messages
                data.append({
                    "message_id": len(data),
                    "text": text,
                    "author_id": str(row.get('author_id', '')),
                    "created_at": str(row.get('created_at', ''))
                })
        
        # Cache the data
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Cached {len(data)} messages")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("📝 Using sample data instead...")
        
        # Fallback sample data
        sample_data = [
            {"message_id": 0, "text": "Your service is absolutely terrible! I've been waiting for 3 hours and nobody has helped me. This is completely unacceptable!"},
            {"message_id": 1, "text": "Thank you so much for the quick response! The issue is now resolved and I'm very happy with the outcome."},
            {"message_id": 2, "text": "The app is not working. I'm getting an error when I try to log in. Can you help?"},
            {"message_id": 3, "text": "I'm not sure if this will solve my problem, but I appreciate your help trying to figure it out."},
            {"message_id": 4, "text": "This is the best customer service I've ever experienced! You guys are amazing and so helpful!"},
            {"message_id": 5, "text": "The product is okay but the delivery was very slow. Not great but not terrible either."},
            {"message_id": 6, "text": "I absolutely hate this! Worst purchase ever. Never buying from you again."},
            {"message_id": 7, "text": "Pretty satisfied with the solution you provided. Thanks for your patience."},
            {"message_id": 8, "text": "The support team was not helpful at all. Still waiting for a real solution."},
            {"message_id": 9, "text": "Excellent work! Problem solved quickly and professionally. Very impressed!"}
        ]
        return sample_data


async def main():
    """Main execution function"""
    print("=" * 80)
    print("🎯 DAG-Based Distributed Sentiment Analysis Client")
    print("=" * 80)
    print()
    
    # Configuration
    FOREMAN_HOST = "localhost"
    FOREMAN_PORT = 9000
    SAMPLE_SIZE = 50  # Number of messages to analyze
    USE_ML = False  # Set to True to enable ML models on workers
    
    print("📋 Configuration:")
    print(f"   Foreman: {FOREMAN_HOST}:{FOREMAN_PORT}")
    print(f"   Sample Size: {SAMPLE_SIZE}")
    print(f"   ML Enhancement: {'Enabled' if USE_ML else 'Disabled (lexicon-only)'}")
    print()
    
    # Load dataset
    messages = await load_dataset(sample_size=SAMPLE_SIZE)
    
    if not messages:
        print("❌ No messages to analyze. Exiting.")
        return
    
    print(f"✅ Loaded {len(messages)} customer messages")
    print()
    
    # Connect to foreman
    print(f"🔌 Connecting to foreman at {FOREMAN_HOST}:{FOREMAN_PORT}...")
    try:
        await connect(FOREMAN_HOST, FOREMAN_PORT)
        print("✅ Connected to foreman!")
        
        # Give a moment for the connection to stabilize
        await asyncio.sleep(0.5)
        
    except Exception as e:
        print(f"❌ Failed to connect to foreman: {e}")
        print("💡 Make sure foreman is running: python tests/run_foreman_simple.py")
        return
    
    print()
    print("=" * 80)
    print("🚀 Starting DAG-based sentiment analysis on distributed workers...")
    print("=" * 80)
    print()
    
    # Prepare input data - batch messages to ensure distribution
    # Create batches to distribute across workers (similar to Monte Carlo approach)
    NUM_BATCHES = 10  # Number of batches to create for distribution
    batch_size = len(messages) // NUM_BATCHES
    if batch_size == 0:
        batch_size = 1
        NUM_BATCHES = len(messages)
    
    input_batches = []
    for i in range(0, len(messages), batch_size):
        batch = []
        for msg in messages[i:i + batch_size]:
            batch.append({
                "text": msg["text"],
                "message_id": msg["message_id"],
                "use_ml": USE_ML
            })
        if batch:  # Only add non-empty batches
            input_batches.append(batch)
    
    print(f"📦 Prepared {len(messages)} messages in {len(input_batches)} batches")
    print(f"   ~{len(messages) // len(input_batches)} messages per batch")
    print(f"💡 Tip: Make sure you have multiple workers running!")
    print("   Run in separate terminals: python tests/run_worker_simple.py")
    print()
    
    # Distribute work across workers
    start_time = time.time()
    
    print(f"📤 Distributing {len(input_batches)} batches to workers...")
    print("⏳ Processing with DAG-based sentiment analysis...\n")
    
    batch_results = await distributed_map(dag_sentiment_analysis_worker, input_batches)
    
    # Flatten results from batches
    results = []
    for batch_result in batch_results:
        if isinstance(batch_result, list):
            results.extend(batch_result)
        else:
            results.append(batch_result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print()
    print("=" * 80)
    print("📊 Analysis Complete - Results Summary")
    print("=" * 80)
    print()
    
    # Parse results (they come back as dict or string representation)
    parsed_results = []
    for r in results:
        if isinstance(r, dict):
            parsed_results.append(r)
        elif isinstance(r, str):
            try:
                # Results might be string representation of dict
                parsed_results.append(eval(r))
            except:
                try:
                    parsed_results.append(json.loads(r))
                except:
                    print(f"Warning: Could not parse result: {r[:100]}")
                    parsed_results.append({"status": "failed", "error": "parse_error"})
        else:
            parsed_results.append({"status": "failed", "error": "unknown_type"})
    
    results = parsed_results
    
    # Analyze results
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"⚡ Average Time per Message: {(total_time / len(messages)):.3f}s")
    print()
    
    if successful:
        # Sentiment distribution
        positive = len([r for r in successful if r.get("sentiment_label") == "POSITIVE"])
        negative = len([r for r in successful if r.get("sentiment_label") == "NEGATIVE"])
        neutral = len([r for r in successful if r.get("sentiment_label") == "NEUTRAL"])
        
        # Worker distribution
        worker_ids = [r.get("worker_id", "unknown") for r in successful]
        unique_workers = len(set(worker_ids))
        
        print("📈 Sentiment Distribution:")
        print(f"   💚 Positive: {positive} ({positive/len(successful)*100:.1f}%)")
        print(f"   🔴 Negative: {negative} ({negative/len(successful)*100:.1f}%)")
        print(f"   🟢 Neutral:  {neutral} ({neutral/len(successful)*100:.1f}%)")
        print()
        
        print("👥 Worker Distribution:")
        print(f"   Unique Workers Used: {unique_workers}")
        if unique_workers == 1:
            print("   ⚠️  WARNING: Only 1 worker processed all tasks!")
            print("   💡 Start more workers: python tests/run_worker_simple.py (in multiple terminals)")
        else:
            print(f"   ✅ Tasks distributed across {unique_workers} workers")
            # Show per-worker stats
            from collections import Counter
            worker_counts = Counter(worker_ids)
            print("   Tasks per worker:")
            for wid, count in sorted(worker_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"      Worker {wid}: {count} tasks")
        print()
        
        # Average confidence and DAG metrics
        avg_confidence = sum(r.get("confidence", 0) for r in successful) / len(successful)
        avg_dag_nodes = sum(r.get("dag_node_count", 0) for r in successful) / len(successful)
        avg_latency = sum(r.get("latency_ms", 0) for r in successful) / len(successful)
        
        print("🎯 DAG Analysis Metrics:")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average DAG Nodes: {avg_dag_nodes:.1f}")
        print(f"   Average Worker Latency: {avg_latency:.1f}ms")
        print()
        
        # Show sample results
        print("=" * 80)
        print("📋 Sample Results with DAG Visualization (First 3)")
        print("=" * 80)
        
        for idx, result in enumerate(successful[:3], 1):
            print()
            print(f"{'=' * 80}")
            print(f"EXAMPLE {idx}")
            print(f"{'=' * 80}")
            print(f"Message ID: {result.get('message_id')}")
            print(f"Text: {result.get('text_preview')}")
            print(f"Sentiment: {result.get('sentiment_label')} (Score: {result.get('sentiment_score')})")
            print(f"Confidence: {result.get('confidence')}")
            print(f"DAG Nodes: {result.get('dag_node_count')}")
            print()
            
            # Show DAG visualization
            if result.get('dag_visualization'):
                print("🌳 DAG Structure:")
                print(result['dag_visualization'])
                print()
            
            # Show sentence breakdown if available
            if result.get('sentence_breakdown'):
                print("📊 Sentence Breakdown:")
                for sent in result['sentence_breakdown'][:2]:  # Show first 2 sentences
                    print(f"  - {sent['text'][:60]}... → {sent['score']}")
                print()
            
            # Show key phrases if available
            if result.get('key_phrases'):
                print("🔑 Key Phrases:")
                for phrase in result['key_phrases'][:3]:  # Show first 3 phrases
                    print(f"  - '{phrase['phrase']}' (modifier: {phrase['modifier']}) → {phrase['score']}")
                print()
            
            print("💡 Advice:")
            for advice in result.get('advice', [])[:3]:  # Show first 3 advice items
                print(f"  • {advice}")
            print("-" * 80)
        
        # Critical messages
        critical = [r for r in successful if r.get("sentiment_score", 0.5) < 0.3]
        if critical:
            print()
            print("=" * 80)
            print(f"🚨 Critical Issues Detected ({len(critical)} messages)")
            print("=" * 80)
            for result in critical[:3]:
                print()
                print(f"Message ID: {result.get('message_id')}")
                print(f"Text: {result.get('text_preview')}")
                print(f"Sentiment Score: {result.get('sentiment_score')}")
                print("🔴 REQUIRES IMMEDIATE ATTENTION")
                print("-" * 80)
    
    # Disconnect
    print()
    print("🔌 Disconnecting from foreman...")
    await disconnect()
    print("✅ Disconnected successfully!")
    print()
    print("=" * 80)
    print("✨ DAG-based sentiment analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
