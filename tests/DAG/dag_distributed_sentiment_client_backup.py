#!/usr/bin/env python3
"""
Customer Support Sentiment Analysis Client - Distributed Processing

This script demonstrates:
1. Loading customer support ticket dataset
2. Distributing sentiment analysis across workers (PC and Android)
3. Detecting sentiment changes and providing advice
4. Aggregating results from distributed workers

Workers analyze customer messages and return sentiment scores,
enabling real-time coaching for support agents.
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


def sentiment_analysis_worker(message_data):
    """
    Worker function to analyze sentiment of customer support messages.
    This runs on PC workers and Android workers.
    
    Uses transformer-based ML model for accurate sentiment analysis.
    
    Args:
        message_data: Dict with 'text' and 'message_id'
    
    Returns:
        Dict with sentiment analysis and advice
    """
    import time
    
    start = time.time()
    
    text = message_data.get('text', '')
    message_id = message_data.get('message_id', 0)
    
    # Handle empty text
    if not text or not text.strip():
        return {
            "message_id": message_id,
            "text_preview": "(empty message)",
            "sentiment_score": 0.5,
            "sentiment_label": "NEUTRAL",
            "confidence": 0.0,
            "positive_signals": 0.0,
            "negative_signals": 0.0,
            "advice": ["Empty message - unable to analyze"],
            "latency_ms": 0,
            "status": "success"
        }
    
    try:
        # Import ML libraries (installed on worker first time)
        from transformers import pipeline
        import warnings
        warnings.filterwarnings('ignore')
        
        # Initialize sentiment analyzer (cached after first use on worker)
        # Using a lightweight but accurate model
        # Use globals to cache analyzer across function calls
        if '_sentiment_analyzer_cache' not in globals():
            print("[Worker] Loading sentiment model (first time only)...)")
            globals()['_sentiment_analyzer_cache'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                framework="pt",  # Explicitly use PyTorch
                device=-1  # CPU only for compatibility
            )
            print("[Worker] Model loaded!")
        
        analyzer = globals()['_sentiment_analyzer_cache']
        
        # Truncate text to model's max length
        text_truncated = text[:512]
        
        # Run ML inference
        result = analyzer(text_truncated)[0]
        
        # Convert model output to 0-1 scale (1 = positive, 0 = negative)
        if result['label'] == 'POSITIVE':
            sentiment_score = (result['score'] + 1) / 2  # Map confidence to upper half
            raw_confidence = result['score']
        else:  # NEGATIVE
            sentiment_score = (1 - result['score']) / 2  # Map confidence to lower half
            raw_confidence = result['score']
        
        confidence = raw_confidence
    
        # Determine sentiment label
        if sentiment_score > 0.65:
            label = "POSITIVE"
        elif sentiment_score < 0.35:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        # Calculate positive/negative signals for reporting
        if result['label'] == 'POSITIVE':
            positive_signals = result['score']
            negative_signals = 1 - result['score']
        else:
            positive_signals = 1 - result['score']
            negative_signals = result['score']
        
        # Generate advice based on sentiment
        advice = []
        if sentiment_score < 0.3:
            advice = [
                "🔴 CRITICAL: Customer is highly dissatisfied",
                "Acknowledge frustration immediately: 'I understand how frustrating this must be'",
                "Show empathy: 'I sincerely apologize for the inconvenience'",
                "Take ownership: 'Let me personally ensure this gets resolved'",
                "Escalate if needed: 'I'd like to involve my supervisor to help immediately'",
                "Provide timeline: 'I will have an answer within [specific time]'"
            ]
        elif sentiment_score < 0.5:
            advice = [
                "🟡 CAUTION: Customer showing dissatisfaction",
                "Be proactive: 'I want to make sure we resolve this completely'",
                "Ask clarifying questions: 'What would make this right for you?'",
                "Validate concerns: 'Your concerns are completely valid'",
                "Show action: 'Here's what I'm doing right now...'",
                "Follow up: 'I'll personally follow up to ensure resolution'"
            ]
        elif sentiment_score < 0.7:
            advice = [
                "🟢 NEUTRAL: Customer is engaged but not satisfied yet",
                "Build rapport: Use their name, show genuine interest",
                "Be clear and concise: Avoid jargon, explain steps simply",
                "Set expectations: 'Here's what will happen next...'",
                "Check understanding: 'Does that make sense? Any questions?'",
                "Stay positive: Maintain professional and helpful tone"
            ]
        else:
            advice = [
                "💚 EXCELLENT: Customer is satisfied",
                "Maintain momentum: Keep the positive tone going",
                "Reinforce value: 'We're always here to help you'",
                "Seek feedback: 'Is there anything else I can help with?'",
                "Show appreciation: 'Thank you for your patience and understanding'",
                "Close positively: 'Feel free to reach out anytime'"
            ]
        
        latency_ms = int((time.time() - start) * 1000)
        
        return {
            "message_id": message_id,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": label,
            "confidence": round(confidence, 3),
            "ml_model_label": result['label'],
            "ml_model_score": round(result['score'], 3),
            "positive_signals": round(positive_signals, 3),
            "negative_signals": round(negative_signals, 3),
            "advice": advice,
            "latency_ms": latency_ms,
            "status": "success"
        }
    
    except Exception as e:
        # Fallback if ML model fails
        import traceback
        error_msg = str(e)
        print(f"[Worker Error] {error_msg}")
        traceback.print_exc()
        
        return {
            "message_id": message_id,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "sentiment_score": 0.5,
            "sentiment_label": "ERROR",
            "confidence": 0.0,
            "positive_signals": 0.0,
            "negative_signals": 0.0,
            "advice": [f"Analysis failed: {error_msg}"],
            "latency_ms": int((time.time() - start) * 1000),
            "status": "failed",
            "error": error_msg
        }


async def load_dataset(sample_size=None):
    """Load customer support dataset using kagglehub with caching."""
    import kagglehub
    import pandas as pd
    
    CACHE_FILE = Path("dataset_cache.json")
    
    # Check cache
    dataset_path = None
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                cached_path = Path(cache.get('dataset_path', ''))
                if cached_path.exists() and list(cached_path.glob("*.csv")):
                    print(f"✅ Using cached dataset from: {cached_path}")
                    dataset_path = cached_path
        except:
            pass
    
    # Download if not cached
    if dataset_path is None:
        print("📥 Downloading customer support dataset from Kaggle...")
        path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
        dataset_path = Path(path)
        print(f"✅ Downloaded to: {dataset_path}")
        
        # Save cache
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump({'dataset_path': str(dataset_path)}, f)
            print(f"💾 Dataset path cached")
        except:
            pass
    
    # Load CSV
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    
    print(f"\n📊 Loading dataset from: {csv_files[0].name}")
    df = pd.read_csv(csv_files[0])
    
    print(f"Total messages: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Find text column
    text_column = None
    for col in ['text', 'message', 'description', 'content', 'ticket_description', 
                'customer_message', 'Ticket Description']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        text_column = df.select_dtypes(include=['object']).columns[0]
    
    print(f"Using column: '{text_column}'")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Using sample of {len(df)} messages")
    
    return df, text_column


def detect_sentiment_changes(results, threshold=0.3):
    """Detect significant sentiment changes between consecutive messages."""
    changes = []
    
    # Sort by message_id
    sorted_results = sorted(results, key=lambda x: x['message_id'])
    
    for i in range(1, len(sorted_results)):
        prev = sorted_results[i-1]
        curr = sorted_results[i]
        
        change = curr['sentiment_score'] - prev['sentiment_score']
        
        if abs(change) >= threshold:
            change_type = "IMPROVEMENT" if change > 0 else "DETERIORATION"
            changes.append({
                'from_msg': prev['message_id'],
                'to_msg': curr['message_id'],
                'change': round(change, 3),
                'change_type': change_type,
                'severity': 'HIGH' if abs(change) > 0.5 else 'MEDIUM',
                'previous_score': prev['sentiment_score'],
                'current_score': curr['sentiment_score']
            })
    
    return changes


def generate_report(results, changes, execution_time):
    """Generate comprehensive sentiment analysis report."""
    print("\n" + "=" * 80)
    print("CUSTOMER SUPPORT SENTIMENT ANALYSIS REPORT")
    print("Distributed Processing with CROWDio Workers")
    print("=" * 80)
    
    # Overall statistics
    total_messages = len(results)
    avg_sentiment = sum(r['sentiment_score'] for r in results) / total_messages
    positive_count = sum(1 for r in results if r['sentiment_label'] == 'POSITIVE')
    negative_count = sum(1 for r in results if r['sentiment_label'] == 'NEGATIVE')
    neutral_count = sum(1 for r in results if r['sentiment_label'] == 'NEUTRAL')
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Total messages analyzed: {total_messages}")
    print(f"Processing time: {execution_time:.2f} seconds")
    print(f"Average sentiment score: {avg_sentiment:.3f}")
    print(f"Positive messages: {positive_count} ({positive_count/total_messages*100:.1f}%)")
    print(f"Neutral messages: {neutral_count} ({neutral_count/total_messages*100:.1f}%)")
    print(f"Negative messages: {negative_count} ({negative_count/total_messages*100:.1f}%)")
    
    # Performance stats
    total_latency = sum(r['latency_ms'] for r in results)
    avg_latency = total_latency / total_messages
    print(f"\n⚡ PERFORMANCE:")
    print(f"Average worker latency: {avg_latency:.1f}ms per message")
    print(f"Messages per second: {total_messages/execution_time:.1f}")
    
    # Sentiment changes
    print(f"\n🔄 SENTIMENT CHANGES DETECTED: {len(changes)}")
    if changes:
        improvements = [c for c in changes if c['change_type'] == 'IMPROVEMENT']
        deteriorations = [c for c in changes if c['change_type'] == 'DETERIORATION']
        print(f"Improvements: {len(improvements)}")
        print(f"Deteriorations: {len(deteriorations)}")
        
        critical = [c for c in changes if c['severity'] == 'HIGH']
        if critical:
            print(f"\n⚠️ CRITICAL CHANGES ({len(critical)}):")
            for change in critical[:5]:
                print(f"\n  Message #{change['from_msg']} → #{change['to_msg']}")
                print(f"  {change['change_type']}: {change['change']:+.3f}")
                print(f"  Score: {change['previous_score']:.3f} → {change['current_score']:.3f}")
    
    # Most negative messages (need immediate attention)
    print(f"\n🔴 MOST NEGATIVE MESSAGES (Immediate Attention Required):")
    most_negative = sorted(results, key=lambda x: x['sentiment_score'])[:3]
    for i, msg in enumerate(most_negative, 1):
        print(f"\n  {i}. Message #{msg['message_id']} - Score: {msg['sentiment_score']:.3f}")
        print(f"     Text: {msg['text_preview']}")
        print(f"     💡 KEY ADVICE:")
        for advice in msg['advice'][:3]:
            print(f"        • {advice}")
    
    # Most positive messages (success examples)
    print(f"\n💚 MOST POSITIVE MESSAGES (Success Examples):")
    most_positive = sorted(results, key=lambda x: x['sentiment_score'], reverse=True)[:3]
    for i, msg in enumerate(most_positive, 1):
        print(f"\n  {i}. Message #{msg['message_id']} - Score: {msg['sentiment_score']:.3f}")
        print(f"     Text: {msg['text_preview']}")
    
    print("\n" + "=" * 80)
    print("✨ KEY TAKEAWAYS:")
    print("1. Train agents on conversations where sentiment deteriorated")
    print("2. Replicate approaches from successful (high sentiment) interactions")
    print("3. Review negative messages immediately for escalation")
    print("4. Use real-time sentiment tracking to coach agents during interactions")
    print("=" * 80 + "\n")


async def main():
    """Main execution flow for distributed sentiment analysis."""
    print("🎯 Customer Support Sentiment Analyzer (Distributed)")
    print("Using CROWDio Workers (PC + Android)\n")
    
    # Configuration
    SAMPLE_SIZE = 200  # Adjust based on needs (None = full dataset)
    FOREMAN_HOST = "localhost"
    FOREMAN_PORT = 9000  # WebSocket port for foreman
    
    try:
        # Load dataset
        print("📂 Loading dataset...")
        df, text_column = await load_dataset(sample_size=SAMPLE_SIZE)
        
        # Prepare tasks for workers
        print(f"\n🔧 Preparing {len(df)} messages for distributed processing...")
        tasks = []
        for idx, row in df.iterrows():
            text = str(row[text_column]) if not pd.isna(row[text_column]) else ""
            tasks.append({
                'text': text,
                'message_id': idx
            })
        
        print(f"✅ Created {len(tasks)} tasks")
        
        # Connect to foreman
        print(f"\n🔌 Connecting to foreman at ws://{FOREMAN_HOST}:{FOREMAN_PORT}...")
        await connect(host=FOREMAN_HOST, port=FOREMAN_PORT)
        print(f"✅ Connected!")
        
        # Distribute work to workers
        print(f"\n🚀 Distributing sentiment analysis to workers...")
        print("Workers will process messages and provide real-time advice...\n")
        
        start_time = time.time()
        
        results = await distributed_map(
            sentiment_analysis_worker,
            tasks
        )
        
        execution_time = time.time() - start_time
        
        print(f"\n✅ Analysis complete in {execution_time:.2f} seconds!")
        
        # Parse string results from workers
        parsed_results = []
        for r in results:
            if isinstance(r, str):
                try:
                    # Results come back as string representation of dict
                    parsed_results.append(eval(r))  # Safe here since it's from our workers
                except:
                    try:
                        parsed_results.append(json.loads(r))
                    except:
                        print(f"⚠️ Could not parse: {r[:100]}")
            elif isinstance(r, dict):
                parsed_results.append(r)
        
        print(f"📊 Parsed {len(parsed_results)} results")
        
        # Filter successful results
        successful_results = [r for r in parsed_results if isinstance(r, dict) and r.get('status') == 'success']
        failed_count = len(parsed_results) - len(successful_results)
        
        print(f"✅ Successfully processed: {len(successful_results)}/{len(parsed_results)}")
        
        if failed_count > 0:
            print(f"⚠️ {failed_count} messages failed to process")
            # Show first few errors
            failed_results = [r for r in parsed_results if isinstance(r, dict) and r.get('status') != 'success']
            for i, fr in enumerate(failed_results[:3]):
                print(f"   Error {i+1}: {fr.get('error', 'Unknown error')}")
        
        if not successful_results:
            print("\n❌ No successful results to analyze!")
            print("💡 Check if workers have required packages: pip install transformers torch")
            await disconnect()
            return
        
        # Detect sentiment changes
        print("\n🔍 Detecting sentiment changes...")
        changes = detect_sentiment_changes(successful_results)
        
        # Generate report
        generate_report(successful_results, changes, execution_time)
        
        # Save results
        output_file = Path("customer_sentiment_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                'results': successful_results,
                'changes': changes,
                'execution_time': execution_time,
                'total_messages': len(successful_results)
            }, f, indent=2)
        print(f"💾 Detailed results saved to: {output_file.absolute()}")
        
        # Disconnect
        await disconnect()
        print("\n✅ Disconnected from foreman")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        await disconnect()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await disconnect()


if __name__ == "__main__":
    # Import pandas here for dataset loading
    import pandas as pd
    asyncio.run(main())
