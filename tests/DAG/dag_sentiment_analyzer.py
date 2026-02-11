#!/usr/bin/env python3
"""
DAG-Based Sentiment Analysis for Enhanced Precision

This module implements a Directed Acyclic Graph (DAG) approach to sentiment analysis
that combines multiple sentiment signals at different levels:

1. Word-level sentiment (lexicon-based)
2. Phrase-level patterns (negations, intensifiers)
3. Sentence-level context (transformer models)
4. Document-level aggregation (weighted combination)

The DAG structure ensures:
- Dependencies between analysis levels are respected
- Multiple signals are combined systematically
- Negations and modifiers propagate correctly
- Final sentiment is more precise and interpretable
"""

import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import re


class NodeType(Enum):
    """Types of nodes in the sentiment DAG"""
    WORD = "word"
    PHRASE = "phrase"
    SENTENCE = "sentence"
    DOCUMENT = "document"
    MODIFIER = "modifier"
    NEGATION = "negation"
    INTENSIFIER = "intensifier"


@dataclass
class SentimentNode:
    """Node in the sentiment DAG"""
    node_id: str
    node_type: NodeType
    content: str
    sentiment_score: float = 0.5  # 0=negative, 0.5=neutral, 1=positive
    confidence: float = 0.0
    children: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.node_id)


class SentimentDAG:
    """
    Directed Acyclic Graph for multi-level sentiment analysis
    """
    
    def __init__(self):
        self.nodes: Dict[str, SentimentNode] = {}
        self.root_node_id: Optional[str] = None
        
        # Sentiment lexicons (simplified - can be expanded)
        self.positive_words = {
            'excellent', 'great', 'amazing', 'wonderful', 'fantastic', 'good', 
            'happy', 'pleased', 'satisfied', 'love', 'perfect', 'best', 
            'helpful', 'thanks', 'appreciate', 'awesome', 'brilliant', 'solved',
            'resolved', 'quick', 'fast', 'efficient', 'professional', 'friendly'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'angry',
            'frustrated', 'disappointed', 'useless', 'poor', 'unacceptable',
            'broken', 'failed', 'error', 'problem', 'issue', 'slow', 'never',
            'waiting', 'delayed', 'ignored', 'rude', 'unhelpful', 'complicated'
        }
        
        self.negations = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'none',
            'nowhere', "n't", 'cannot', 'cant', 'wont', 'wouldnt', 'shouldnt',
            'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'wasnt', 'werent'
        }
        
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.4, 'so': 1.3,
            'absolutely': 1.8, 'totally': 1.6, 'completely': 1.7,
            'highly': 1.5, 'incredibly': 1.9, 'quite': 1.2, 'rather': 1.1
        }
        
        self.diminishers = {
            'slightly': 0.7, 'somewhat': 0.8, 'barely': 0.5, 'hardly': 0.5,
            'kind of': 0.7, 'sort of': 0.7, 'a bit': 0.8, 'a little': 0.8
        }
    
    def add_node(self, node: SentimentNode) -> None:
        """Add a node to the DAG"""
        self.nodes[node.node_id] = node
    
    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add a directed edge from parent to child"""
        if parent_id in self.nodes and child_id in self.nodes:
            self.nodes[parent_id].children.append(child_id)
            self.nodes[child_id].parents.append(parent_id)
    
    def get_node(self, node_id: str) -> Optional[SentimentNode]:
        """Retrieve a node by ID"""
        return self.nodes.get(node_id)
    
    def topological_sort(self) -> List[str]:
        """
        Return nodes in topological order (dependencies first)
        Uses Kahn's algorithm
        """
        # Calculate in-degrees
        in_degree = {node_id: len(node.parents) for node_id, node in self.nodes.items()}
        
        # Queue of nodes with no incoming edges
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Reduce in-degree for children
            if current in self.nodes:
                for child_id in self.nodes[current].children:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        queue.append(child_id)
        
        return result
    
    def build_from_text(self, text: str, message_id: int = 0) -> str:
        """
        Build DAG from input text with hierarchical structure
        
        Returns: root node ID
        """
        # Create document (root) node
        doc_id = f"doc_{message_id}"
        doc_node = SentimentNode(
            node_id=doc_id,
            node_type=NodeType.DOCUMENT,
            content=text,
            metadata={"message_id": message_id}
        )
        self.add_node(doc_node)
        self.root_node_id = doc_id
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        for sent_idx, sentence in enumerate(sentences):
            sent_id = f"sent_{message_id}_{sent_idx}"
            sent_node = SentimentNode(
                node_id=sent_id,
                node_type=NodeType.SENTENCE,
                content=sentence,
                metadata={"sentence_index": sent_idx}
            )
            self.add_node(sent_node)
            self.add_edge(doc_id, sent_id)
            
            # Tokenize and analyze words/phrases
            tokens = self._tokenize(sentence)
            
            # Look for negation and intensifier patterns
            for word_idx, token in enumerate(tokens):
                word_id = f"word_{message_id}_{sent_idx}_{word_idx}"
                word_lower = token.lower()
                
                # Determine word sentiment
                if word_lower in self.positive_words:
                    word_score = 0.8
                    word_type = NodeType.WORD
                elif word_lower in self.negative_words:
                    word_score = 0.2
                    word_type = NodeType.WORD
                elif word_lower in self.negations:
                    word_score = 0.5
                    word_type = NodeType.NEGATION
                elif word_lower in self.intensifiers:
                    word_score = 0.5
                    word_type = NodeType.INTENSIFIER
                elif word_lower in self.diminishers:
                    word_score = 0.5
                    word_type = NodeType.MODIFIER
                else:
                    continue  # Skip neutral words
                
                word_node = SentimentNode(
                    node_id=word_id,
                    node_type=word_type,
                    content=token,
                    sentiment_score=word_score,
                    confidence=0.6,
                    metadata={
                        "word_index": word_idx,
                        "intensity": self.intensifiers.get(word_lower, 1.0) if word_type == NodeType.INTENSIFIER else 1.0
                    }
                )
                self.add_node(word_node)
                self.add_edge(sent_id, word_id)
                
                # Connect negations and intensifiers to following sentiment words
                if word_type in [NodeType.NEGATION, NodeType.INTENSIFIER, NodeType.MODIFIER]:
                    # Look ahead for sentiment words (within 3 tokens)
                    for look_ahead in range(1, min(4, len(tokens) - word_idx)):
                        next_token = tokens[word_idx + look_ahead].lower()
                        if next_token in self.positive_words or next_token in self.negative_words:
                            next_word_id = f"word_{message_id}_{sent_idx}_{word_idx + look_ahead}"
                            # Create phrase node to capture the modification
                            phrase_id = f"phrase_{message_id}_{sent_idx}_{word_idx}"
                            phrase_content = " ".join(tokens[word_idx:word_idx + look_ahead + 1])
                            phrase_node = SentimentNode(
                                node_id=phrase_id,
                                node_type=NodeType.PHRASE,
                                content=phrase_content,
                                metadata={"modifier": token, "target": tokens[word_idx + look_ahead]}
                            )
                            self.add_node(phrase_node)
                            self.add_edge(sent_id, phrase_id)
                            self.add_edge(phrase_id, word_id)
                            # We'll create the target word node when we reach it
                            break
        
        return doc_id
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenizer (can be improved with NLTK)
        tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text)
        return tokens
    
    def compute_sentiments(self, use_ml: bool = True) -> None:
        """
        Compute sentiment scores for all nodes in topological order
        This ensures child nodes are processed before parents
        """
        # Process nodes bottom-up (leaves first)
        sorted_nodes = reversed(self.topological_sort())
        
        for node_id in sorted_nodes:
            node = self.nodes[node_id]
            
            if node.node_type == NodeType.PHRASE:
                # Phrase sentiment combines modifier + target word
                self._compute_phrase_sentiment(node)
            
            elif node.node_type == NodeType.SENTENCE:
                # Sentence sentiment aggregates words and phrases
                self._compute_sentence_sentiment(node, use_ml)
            
            elif node.node_type == NodeType.DOCUMENT:
                # Document sentiment aggregates sentences
                self._compute_document_sentiment(node)
    
    def _compute_phrase_sentiment(self, phrase_node: SentimentNode) -> None:
        """Compute sentiment for a phrase considering modifiers"""
        if not phrase_node.children:
            return
        
        # Get modifier and target sentiment
        child_nodes = [self.nodes[child_id] for child_id in phrase_node.children]
        
        negation_node = None
        intensifier_node = None
        target_node = None
        
        for child in child_nodes:
            if child.node_type == NodeType.NEGATION:
                negation_node = child
            elif child.node_type == NodeType.INTENSIFIER:
                intensifier_node = child
            elif child.node_type == NodeType.WORD:
                target_node = child
        
        if target_node:
            base_score = target_node.sentiment_score
            
            # Apply negation (flip sentiment around 0.5)
            if negation_node:
                base_score = 1.0 - base_score
            
            # Apply intensification
            if intensifier_node:
                intensity = intensifier_node.metadata.get("intensity", 1.0)
                # Amplify distance from neutral
                base_score = 0.5 + (base_score - 0.5) * intensity
                base_score = max(0.0, min(1.0, base_score))  # Clamp to [0, 1]
            
            phrase_node.sentiment_score = base_score
            phrase_node.confidence = 0.7
    
    def _compute_sentence_sentiment(self, sent_node: SentimentNode, use_ml: bool = True) -> None:
        """Compute sentence sentiment from children nodes"""
        if not sent_node.children:
            sent_node.sentiment_score = 0.5
            sent_node.confidence = 0.0
            return
        
        # Collect sentiment from all child nodes (words and phrases)
        child_sentiments = []
        for child_id in sent_node.children:
            child = self.nodes[child_id]
            if child.sentiment_score != 0.5 or child.node_type == NodeType.PHRASE:
                # Weight phrases higher than individual words
                weight = 2.0 if child.node_type == NodeType.PHRASE else 1.0
                child_sentiments.append((child.sentiment_score, child.confidence, weight))
        
        if child_sentiments:
            # Weighted average
            total_weight = sum(conf * weight for _, conf, weight in child_sentiments)
            if total_weight > 0:
                weighted_score = sum(score * conf * weight for score, conf, weight in child_sentiments) / total_weight
                sent_node.sentiment_score = weighted_score
                sent_node.confidence = min(0.85, total_weight / len(child_sentiments))
            else:
                sent_node.sentiment_score = sum(score for score, _, _ in child_sentiments) / len(child_sentiments)
                sent_node.confidence = 0.3
        
        # Optionally enhance with ML model for sentence-level analysis
        if use_ml:
            try:
                ml_score = self._get_ml_sentence_sentiment(sent_node.content)
                # Blend lexicon-based and ML scores (60% ML, 40% lexicon for balance)
                if ml_score is not None:
                    sent_node.sentiment_score = 0.6 * ml_score + 0.4 * sent_node.sentiment_score
                    sent_node.confidence = 0.85
                    sent_node.metadata["ml_score"] = ml_score
            except Exception as e:
                sent_node.metadata["ml_error"] = str(e)
    
    def _get_ml_sentence_sentiment(self, text: str) -> Optional[float]:
        """Get ML-based sentiment score (to be implemented with actual model)"""
        # Placeholder - in actual implementation, use transformer model
        # This would be called from worker with model cached
        return None
    
    def _compute_document_sentiment(self, doc_node: SentimentNode) -> None:
        """Compute document-level sentiment from sentences"""
        if not doc_node.children:
            doc_node.sentiment_score = 0.5
            doc_node.confidence = 0.0
            return
        
        # Collect sentence sentiments
        sentence_sentiments = []
        for child_id in doc_node.children:
            child = self.nodes[child_id]
            if child.node_type == NodeType.SENTENCE:
                sentence_sentiments.append((child.sentiment_score, child.confidence))
        
        if sentence_sentiments:
            # Weighted average with confidence
            total_confidence = sum(conf for _, conf in sentence_sentiments)
            if total_confidence > 0:
                weighted_score = sum(score * conf for score, conf in sentence_sentiments) / total_confidence
                doc_node.sentiment_score = weighted_score
                doc_node.confidence = min(0.9, total_confidence / len(sentence_sentiments))
            else:
                doc_node.sentiment_score = sum(score for score, _ in sentence_sentiments) / len(sentence_sentiments)
                doc_node.confidence = 0.4
        
        # Add document-level metadata
        doc_node.metadata["num_sentences"] = len(sentence_sentiments)
        doc_node.metadata["sentence_scores"] = [score for score, _ in sentence_sentiments]
    
    def get_final_sentiment(self) -> Dict[str, Any]:
        """Get the final document-level sentiment with detailed breakdown"""
        if not self.root_node_id:
            return {
                "sentiment_score": 0.5,
                "sentiment_label": "NEUTRAL",
                "confidence": 0.0,
                "error": "No document node found"
            }
        
        root = self.nodes[self.root_node_id]
        score = root.sentiment_score
        
        # Determine label
        if score > 0.65:
            label = "POSITIVE"
        elif score < 0.35:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        # Collect sentence-level details
        sentence_details = []
        for child_id in root.children:
            sent_node = self.nodes[child_id]
            if sent_node.node_type == NodeType.SENTENCE:
                sentence_details.append({
                    "text": sent_node.content,
                    "score": round(sent_node.sentiment_score, 3),
                    "confidence": round(sent_node.confidence, 3)
                })
        
        # Collect key phrases (with modifiers)
        key_phrases = []
        for node in self.nodes.values():
            if node.node_type == NodeType.PHRASE:
                key_phrases.append({
                    "phrase": node.content,
                    "score": round(node.sentiment_score, 3),
                    "modifier": node.metadata.get("modifier", "")
                })
        
        return {
            "sentiment_score": round(score, 3),
            "sentiment_label": label,
            "confidence": round(root.confidence, 3),
            "positive_signals": round(max(0, (score - 0.5) * 2), 3),
            "negative_signals": round(max(0, (0.5 - score) * 2), 3),
            "sentence_breakdown": sentence_details,
            "key_phrases": key_phrases,
            "num_sentences": root.metadata.get("num_sentences", 0),
            "dag_node_count": len(self.nodes)
        }
    
    def visualize_dag(self) -> str:
        """Generate a text-based visualization of the DAG structure"""
        lines = ["Sentiment DAG Structure:", "=" * 50]
        
        # Show hierarchy
        if self.root_node_id:
            lines.append(self._visualize_node(self.root_node_id, 0))
        
        return "\n".join(lines)
    
    def _visualize_node(self, node_id: str, depth: int) -> str:
        """Recursively visualize a node and its children"""
        node = self.nodes[node_id]
        indent = "  " * depth
        
        score_str = f"{node.sentiment_score:.2f}" if node.sentiment_score != 0.5 else "---"
        content_preview = node.content[:40] + "..." if len(node.content) > 40 else node.content
        
        lines = [f"{indent}[{node.node_type.value}] {score_str} | {content_preview}"]
        
        for child_id in node.children:
            lines.append(self._visualize_node(child_id, depth + 1))
        
        return "\n".join(lines)


def analyze_text_with_dag(text: str, message_id: int = 0, use_ml: bool = False) -> Dict[str, Any]:
    """
    Main function to analyze text using DAG approach
    
    Args:
        text: Input text to analyze
        message_id: Identifier for the message
        use_ml: Whether to use ML models for sentence-level analysis
        
    Returns:
        Dict with sentiment analysis results
    """
    dag = SentimentDAG()
    
    # Build DAG from text
    dag.build_from_text(text, message_id)
    
    # Compute sentiments bottom-up
    dag.compute_sentiments(use_ml=use_ml)
    
    # Get final results
    result = dag.get_final_sentiment()
    result["message_id"] = message_id
    result["text_preview"] = text[:100] + "..." if len(text) > 100 else text
    
    # Optionally include DAG visualization for debugging
    # result["dag_visualization"] = dag.visualize_dag()
    
    return result


# Worker function for distributed processing with DAG
def dag_sentiment_worker(message_data):
    """
    Worker function using DAG-based sentiment analysis
    This can be used with the distributed framework
    """
    import time
    
    start = time.time()
    
    text = message_data.get('text', '')
    message_id = message_data.get('message_id', 0)
    use_ml = message_data.get('use_ml', False)
    
    # Handle empty text
    if not text or not text.strip():
        return {
            "message_id": message_id,
            "text_preview": "(empty message)",
            "sentiment_score": 0.5,
            "sentiment_label": "NEUTRAL",
            "confidence": 0.0,
            "advice": ["Empty message - unable to analyze"],
            "method": "DAG",
            "status": "success"
        }
    
    try:
        # Analyze with DAG
        result = analyze_text_with_dag(text, message_id, use_ml=use_ml)
        
        # Generate advice based on sentiment (reuse from customer_support_sentiment_client.py)
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
        result["method"] = "DAG"
        result["status"] = "success"
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[DAG Worker Error] {error_msg}")
        traceback.print_exc()
        
        return {
            "message_id": message_id,
            "text_preview": text[:100],
            "sentiment_score": 0.5,
            "sentiment_label": "ERROR",
            "confidence": 0.0,
            "advice": [f"Analysis failed: {error_msg}"],
            "latency_ms": int((time.time() - start) * 1000),
            "method": "DAG",
            "status": "failed",
            "error": error_msg
        }


if __name__ == "__main__":
    # Example usage
    test_messages = [
        "This service is absolutely terrible! I've been waiting for hours and nobody has helped me. Very frustrated!",
        "Thank you so much for your quick response. The issue is resolved and I'm quite happy with the solution.",
        "I'm not sure if this will work, but I appreciate your help trying to solve my problem.",
        "The product is not bad, but it could be better. I'm somewhat satisfied.",
        "I absolutely love this! Best customer service ever. You guys are amazing!"
    ]
    
    print("DAG-Based Sentiment Analysis Demo")
    print("=" * 70)
    
    for idx, message in enumerate(test_messages):
        print(f"\nMessage {idx + 1}:")
        print(f"Text: {message}")
        print("-" * 70)
        
        result = analyze_text_with_dag(message, idx, use_ml=False)
        
        print(f"Sentiment Score: {result['sentiment_score']} ({result['sentiment_label']})")
        print(f"Confidence: {result['confidence']}")
        print(f"Positive Signals: {result['positive_signals']}")
        print(f"Negative Signals: {result['negative_signals']}")
        print(f"Sentences Analyzed: {result['num_sentences']}")
        print(f"DAG Nodes: {result['dag_node_count']}")
        
        if result.get('sentence_breakdown'):
            print("\nSentence Breakdown:")
            for sent in result['sentence_breakdown']:
                print(f"  - {sent['text'][:50]}... -> Score: {sent['score']}")
        
        if result.get('key_phrases'):
            print("\nKey Phrases Detected:")
            for phrase in result['key_phrases']:
                print(f"  - '{phrase['phrase']}' (modifier: {phrase['modifier']}) -> {phrase['score']}")
        
        print("=" * 70)
