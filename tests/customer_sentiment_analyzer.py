"""
Customer Support Sentiment Analyzer
Analyzes customer sentiment over conversation, detects changes, and provides advice to support agents.
"""

import kagglehub
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline
import torch
from typing import List, Dict, Tuple
import json
import warnings
warnings.filterwarnings('ignore')


class CustomerSentimentAnalyzer:
    """Analyzes customer sentiment and provides actionable advice for support agents."""
    
    CACHE_FILE = Path("dataset_cache.json")
    
    def __init__(self):
        """Initialize the sentiment analyzer with a pre-trained model."""
        print("Loading sentiment analysis model...")
        # Use a robust sentiment model with PyTorch framework
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",  # Explicitly use PyTorch
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"Model loaded. Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
    def download_dataset(self) -> Path:
        """Download the customer support ticket dataset from Kaggle (cached)."""
        # Check cache first
        if self.CACHE_FILE.exists():
            try:
                with open(self.CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                    cached_path = Path(cache.get('dataset_path', ''))
                    
                    # Verify the cached path still exists and has CSV files
                    if cached_path.exists() and list(cached_path.glob("*.csv")):
                        print(f"✅ Using cached dataset from: {cached_path}")
                        return cached_path
                    else:
                        print("⚠️ Cached path no longer valid, re-downloading...")
            except (json.JSONDecodeError, KeyError):
                print("⚠️ Cache file corrupted, re-downloading...")
        
        # Download dataset
        print("📥 Downloading customer support ticket dataset from Kaggle...")
        path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
        print(f"✅ Downloaded to: {path}")
        
        # Save to cache
        try:
            with open(self.CACHE_FILE, 'w') as f:
                json.dump({'dataset_path': str(path)}, f)
            print(f"💾 Dataset path cached to {self.CACHE_FILE}")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
        
        return Path(path)
    
    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Load and explore the dataset."""
        # Find CSV files in the dataset
        csv_files = list(dataset_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
        
        print(f"\nFound CSV file: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
        
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:\n{df.head()}")
        
        return df
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text."""
        if pd.isna(text) or not text.strip():
            return {"label": "NEUTRAL", "score": 0.5}
        
        # Truncate text to avoid token limits
        text = text[:512]
        result = self.sentiment_analyzer(text)[0]
        
        # Convert to numerical score: POSITIVE = 1.0, NEGATIVE = 0.0
        score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
        
        return {
            "label": result['label'],
            "score": score,
            "sentiment_value": score  # 0-1 scale where 1 is most positive
        }
    
    def analyze_conversation(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Analyze sentiment for all messages in the dataset."""
        print(f"\nAnalyzing sentiment for {len(df)} messages...")
        
        sentiments = []
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{len(df)}")
            
            sentiment = self.analyze_sentiment(str(text))
            sentiments.append(sentiment)
        
        # Add sentiment data to dataframe
        df['sentiment_label'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['sentiment_value'] for s in sentiments]
        
        return df
    
    def detect_sentiment_changes(self, df: pd.DataFrame, threshold: float = 0.3) -> List[Dict]:
        """
        Detect significant sentiment changes in the conversation.
        
        Args:
            df: DataFrame with sentiment scores
            threshold: Minimum change in sentiment to be considered significant
        
        Returns:
            List of sentiment change events
        """
        changes = []
        
        if len(df) < 2:
            return changes
        
        for i in range(1, len(df)):
            prev_score = df.iloc[i-1]['sentiment_score']
            curr_score = df.iloc[i]['sentiment_score']
            change = curr_score - prev_score
            
            if abs(change) >= threshold:
                change_type = "IMPROVEMENT" if change > 0 else "DETERIORATION"
                changes.append({
                    'index': i,
                    'previous_score': prev_score,
                    'current_score': curr_score,
                    'change': change,
                    'change_type': change_type,
                    'severity': 'HIGH' if abs(change) > 0.5 else 'MEDIUM'
                })
        
        return changes
    
    def get_advice(self, sentiment_score: float, change_type: str = None) -> List[str]:
        """
        Provide actionable advice based on sentiment analysis.
        
        Args:
            sentiment_score: Current sentiment score (0-1)
            change_type: Type of change (IMPROVEMENT, DETERIORATION, or None)
        
        Returns:
            List of advice strings
        """
        advice = []
        
        # General advice based on current sentiment
        if sentiment_score < 0.3:  # Negative sentiment
            advice.extend([
                "🔴 CRITICAL: Customer is highly dissatisfied",
                "• Acknowledge their frustration immediately: 'I understand how frustrating this must be for you'",
                "• Show empathy: 'I sincerely apologize for the inconvenience'",
                "• Take ownership: 'Let me personally ensure this gets resolved'",
                "• Offer escalation: 'I'd like to escalate this to my supervisor to get you immediate help'",
                "• Provide timeline: 'I will have an answer for you within [specific time]'",
            ])
        elif sentiment_score < 0.5:  # Somewhat negative
            advice.extend([
                "🟡 CAUTION: Customer is showing signs of dissatisfaction",
                "• Be proactive: 'I want to make sure we resolve this to your satisfaction'",
                "• Ask clarifying questions: 'Can you help me understand what would make this right?'",
                "• Validate concerns: 'Your concerns are completely valid'",
                "• Show action: 'Here's what I'm going to do right now...'",
            ])
        elif sentiment_score < 0.7:  # Neutral
            advice.extend([
                "🟢 NEUTRAL: Customer is engaged but not particularly satisfied",
                "• Build rapport: Use their name, show genuine interest",
                "• Be clear and concise: Avoid jargon, explain steps clearly",
                "• Set expectations: 'Here's what will happen next...'",
                "• Check understanding: 'Does that make sense? Any questions?'",
            ])
        else:  # Positive sentiment
            advice.extend([
                "💚 EXCELLENT: Customer is satisfied",
                "• Maintain momentum: Keep the positive tone",
                "• Reinforce value: 'We're always here to help'",
                "• Seek feedback: 'Is there anything else I can help you with?'",
                "• Show appreciation: 'Thank you for your patience/understanding'",
            ])
        
        # Specific advice based on sentiment change
        if change_type == "DETERIORATION":
            advice.extend([
                "\n⚠️ SENTIMENT DETERIORATING:",
                "• Stop and reassess: 'Let me make sure I understand your main concern'",
                "• Change approach: Current strategy isn't working",
                "• Increase empathy: 'I can hear that you're frustrated'",
                "• Offer alternatives: 'Let me suggest a different solution'",
                "• Consider transfer: 'Would it help to speak with a specialist?'",
            ])
        elif change_type == "IMPROVEMENT":
            advice.extend([
                "\n✅ SENTIMENT IMPROVING:",
                "• Continue current approach: What you're doing is working",
                "• Build on progress: 'I'm glad we're making progress together'",
                "• Stay consistent: Maintain tone and pace",
                "• Move toward resolution: 'Let's wrap this up for you'",
            ])
        
        return advice
    
    def generate_report(self, df: pd.DataFrame, changes: List[Dict], 
                       text_column: str, ticket_id_column: str = None) -> str:
        """Generate a comprehensive sentiment analysis report."""
        report = []
        report.append("=" * 80)
        report.append("CUSTOMER SENTIMENT ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Overall statistics
        report.append(f"\n📊 OVERALL STATISTICS:")
        report.append(f"Total messages analyzed: {len(df)}")
        report.append(f"Average sentiment score: {df['sentiment_score'].mean():.2f}")
        report.append(f"Positive messages: {(df['sentiment_label'] == 'POSITIVE').sum()} ({(df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100:.1f}%)")
        report.append(f"Negative messages: {(df['sentiment_label'] == 'NEGATIVE').sum()} ({(df['sentiment_label'] == 'NEGATIVE').sum() / len(df) * 100:.1f}%)")
        
        # Sentiment distribution
        report.append(f"\n📈 SENTIMENT DISTRIBUTION:")
        report.append(f"Highly Positive (>0.7): {(df['sentiment_score'] > 0.7).sum()}")
        report.append(f"Neutral (0.3-0.7): {((df['sentiment_score'] >= 0.3) & (df['sentiment_score'] <= 0.7)).sum()}")
        report.append(f"Highly Negative (<0.3): {(df['sentiment_score'] < 0.3).sum()}")
        
        # Sentiment changes
        report.append(f"\n🔄 SENTIMENT CHANGES DETECTED: {len(changes)}")
        if changes:
            deteriorations = [c for c in changes if c['change_type'] == 'DETERIORATION']
            improvements = [c for c in changes if c['change_type'] == 'IMPROVEMENT']
            report.append(f"Improvements: {len(improvements)}")
            report.append(f"Deteriorations: {len(deteriorations)}")
            
            # Show critical changes
            critical_changes = [c for c in changes if c['severity'] == 'HIGH']
            if critical_changes:
                report.append(f"\n⚠️ CRITICAL CHANGES ({len(critical_changes)}):")
                for change in critical_changes[:5]:  # Show first 5
                    report.append(f"\nMessage #{change['index']}:")
                    report.append(f"  Change: {change['change']:.2f} ({change['change_type']})")
                    report.append(f"  Previous: {change['previous_score']:.2f} → Current: {change['current_score']:.2f}")
                    if change['index'] < len(df):
                        report.append(f"  Text: {df.iloc[change['index']][text_column][:100]}...")
        
        # Sample messages by sentiment
        report.append(f"\n💬 SAMPLE MESSAGES:")
        
        # Most negative
        most_negative = df.nsmallest(3, 'sentiment_score')
        report.append(f"\nMost Negative Messages:")
        for idx, row in most_negative.iterrows():
            report.append(f"\n  Score: {row['sentiment_score']:.2f}")
            report.append(f"  Text: {row[text_column][:150]}...")
            advice = self.get_advice(row['sentiment_score'])
            report.append(f"  💡 ADVICE:")
            for tip in advice[:4]:  # Show first 4 tips
                report.append(f"     {tip}")
        
        # Most positive
        most_positive = df.nlargest(3, 'sentiment_score')
        report.append(f"\nMost Positive Messages:")
        for idx, row in most_positive.iterrows():
            report.append(f"\n  Score: {row['sentiment_score']:.2f}")
            report.append(f"  Text: {row[text_column][:150]}...")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def analyze_and_report(self, sample_size: int = None):
        """Main method to run the complete analysis."""
        # Download and load dataset
        dataset_path = self.download_dataset()
        df = self.load_dataset(dataset_path)
        
        # Use sample if requested
        if sample_size and sample_size < len(df):
            print(f"\nUsing sample of {sample_size} messages")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Determine text column (try common names)
        text_column = None
        for col in ['text', 'message', 'description', 'content', 'ticket_description', 'customer_message']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            # Use first text column
            text_column = df.select_dtypes(include=['object']).columns[0]
        
        print(f"\nUsing '{text_column}' as text column")
        
        # Analyze sentiment
        df = self.analyze_conversation(df, text_column)
        
        # Detect changes
        changes = self.detect_sentiment_changes(df)
        
        # Generate report
        report = self.generate_report(df, changes, text_column)
        print("\n" + report)
        
        # Save results
        output_file = Path("customer_sentiment_results.csv")
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file.absolute()}")
        
        # Save report
        report_file = Path("customer_sentiment_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file.absolute()}")
        
        return df, changes


def main():
    """Run the customer sentiment analyzer."""
    print("🎯 Customer Support Sentiment Analyzer")
    print("=" * 80)
    
    analyzer = CustomerSentimentAnalyzer()
    
    # Analyze with sample (set to None to analyze full dataset)
    # Using sample for faster testing
    df, changes = analyzer.analyze_and_report(sample_size=500)
    
    print("\n✨ Analysis complete!")
    print("\nKey Takeaways:")
    print("1. Review messages with sentiment scores < 0.3 for immediate attention")
    print("2. Train agents on conversations where sentiment deteriorated")
    print("3. Replicate approaches from conversations where sentiment improved")
    print("4. Use the advice provided to coach agents on handling different sentiment levels")


if __name__ == "__main__":
    main()
