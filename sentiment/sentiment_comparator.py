"""
Sentiment analysis comparator for testing different preprocessing + embedding + model combinations.

This module allows testing various combinations of:
- Preprocessing: basic vs light
- Embeddings: BERT vs FastText  
- Models: VADER vs BERT sentiment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Import preprocessing modules
from preprocessing.basic.text_cleaner import TextCleaner as BasicCleaner
from preprocessing.basic.stopword_remover import StopwordRemover as BasicStopwordRemover
from preprocessing.basic.text_lemmatizer import TextLemmatizer as BasicLemmatizer

from preprocessing.light.text_cleaner_light import TextCleaner as LightCleaner
from preprocessing.light.stopword_remover_light import StopwordRemover as LightStopwordRemover
from preprocessing.light.text_lemmatizer_light import TextLemmatizer as LightLemmatizer

# Import embedding modules
from embedding.bert import Bert
from embedding.fasttext import FastText

# Import sentiment modules
from sentiment.vader_sentiment import VaderSentiment
from sentiment.bert_sentiment import BertSentiment


class SentimentComparator:
	"""
	Comparator for testing different preprocessing + embedding + sentiment model combinations.
	
	Tests combinations of:
	- Preprocessing: basic vs light
	- Embeddings: BERT vs FastText (note: VADER doesn't use embeddings)
	- Models: VADER vs BERT sentiment
	"""
	
	def __init__(self):
		"""Initialize the comparator."""
		self.results = {}
		self.sample_texts = [
			"I love this movie! It's absolutely amazing and fantastic!",
			"This is terrible, I hate it so much. Worst experience ever.",
			"The weather is okay today, nothing special really.",
			"AMAZING product! Best purchase I've ever made! So happy!",
			"Complete waste of money. Don't buy this garbage.",
			"It's fine, does what it's supposed to do.",
			"Absolutely brilliant! Can't recommend it enough!",
			"Meh, it's average at best. Nothing to write home about."
		]
		
		# Expected sentiment labels for evaluation (manual labeling)
		self.expected_labels = [
			'positive',  # "I love this movie..."
			'negative',  # "This is terrible..."
			'neutral',   # "The weather is okay..."
			'positive',  # "AMAZING product..."
			'negative',  # "Complete waste..."
			'neutral',   # "It's fine..."
			'positive',  # "Absolutely brilliant..."
			'neutral'    # "Meh, it's average..."
		]
	
	def preprocess_texts(self, texts: List[str], preprocessing_type: str = 'basic') -> List[List[str]]:
		"""
		Preprocess texts using specified preprocessing approach.
		
		Args:
			texts: List of raw texts
			preprocessing_type: 'basic' or 'light'
			
		Returns:
			List of preprocessed tokenized texts
		"""
		if preprocessing_type == 'basic':
			cleaner = BasicCleaner()
			stopword_remover = BasicStopwordRemover()
			lemmatizer = BasicLemmatizer()
		else:  # light
			cleaner = LightCleaner()
			stopword_remover = LightStopwordRemover()
			lemmatizer = LightLemmatizer()
		
		# Apply preprocessing pipeline
		processed_texts = []
		for text in texts:
			# Clean text
			cleaned = cleaner.clean_text(text)
			
			# Tokenize (simple word splitting for now)
			tokens = cleaned.split()
			
			# Remove stopwords
			tokens = stopword_remover.remove_stopwords(tokens)
			
			# Lemmatize
			tokens = lemmatizer.lemmatize(tokens)
			
			processed_texts.append(tokens)
		
		return processed_texts
	
	def test_combination(self, 
						preprocessing_type: str, 
						embedding_type: str, 
						model_type: str,
						texts: List[str] = None) -> Dict[str, Any]:
		"""
		Test a specific combination of preprocessing + embedding + model.
		
		Args:
			preprocessing_type: 'basic' or 'light'
			embedding_type: 'bert' or 'fasttext' (not used for VADER)
			model_type: 'vader' or 'bert'
			texts: List of texts to analyze (uses sample texts if None)
			
		Returns:
			Dictionary with results and metrics
		"""
		if texts is None:
			texts = self.sample_texts
		
		print(f"\n🔍 Testing: {preprocessing_type.upper()} preprocessing + {embedding_type.upper()} embedding + {model_type.upper()} model")
		
		start_time = time.time()
		
		# Step 1: Preprocessing
		print("  📝 Preprocessing texts...")
		processed_texts = self.preprocess_texts(texts, preprocessing_type)
		
		# Step 2: Sentiment Analysis
		print("  🎯 Analyzing sentiment...")
		if model_type == 'vader':
			# VADER doesn't use embeddings, works directly with text
			model = VaderSentiment()
			predictions = model.fit_predict(processed_texts)
			
		else:  # bert sentiment
			# BERT sentiment model also doesn't use separate embeddings
			# It uses its own internal embeddings
			model = BertSentiment()
			predictions = model.fit_predict(processed_texts)
		
		end_time = time.time()
		processing_time = end_time - start_time
		
		# Step 3: Evaluate results
		predicted_labels = predictions['classification']
		
		# Calculate metrics
		accuracy = accuracy_score(self.expected_labels, predicted_labels)
		
		# Prepare results
		results = {
			'preprocessing': preprocessing_type,
			'embedding': embedding_type,
			'model': model_type,
			'predictions': predictions,
			'predicted_labels': predicted_labels,
			'expected_labels': self.expected_labels,
			'accuracy': accuracy,
			'processing_time': processing_time,
			'texts': texts,
			'processed_texts': processed_texts
		}
		
		print(f"  ✅ Accuracy: {accuracy:.3f} | Time: {processing_time:.2f}s")
		
		return results
	
	def run_all_combinations(self, texts: List[str] = None) -> Dict[str, Dict[str, Any]]:
		"""
		Run all combinations of preprocessing + embedding + model.
		
		Args:
			texts: List of texts to analyze (uses sample texts if None)
			
		Returns:
			Dictionary with all results
		"""
		if texts is None:
			texts = self.sample_texts
		
		print("🚀 Starting comprehensive sentiment analysis comparison...")
		
		combinations = []
		
		# VADER combinations (doesn't really use embeddings, but we test for completeness)
		combinations.extend([
			('basic', 'bert', 'vader'),
			('basic', 'fasttext', 'vader'),
			('light', 'bert', 'vader'),
			('light', 'fasttext', 'vader'),
		])
		
		# BERT sentiment combinations  
		combinations.extend([
			('basic', 'bert', 'bert'),
			('basic', 'fasttext', 'bert'),
			('light', 'bert', 'bert'),
			('light', 'fasttext', 'bert'),
		])
		
		results = {}
		
		for i, (prep, emb, model) in enumerate(combinations, 1):
			print(f"\n📊 Combination {i}/{len(combinations)}")
			
			combination_name = f"{prep}_{emb}_{model}"
			
			try:
				result = self.test_combination(prep, emb, model, texts)
				results[combination_name] = result
				
			except Exception as e:
				print(f"  ❌ Error: {str(e)}")
				results[combination_name] = {
					'error': str(e),
					'preprocessing': prep,
					'embedding': emb,
					'model': model
				}
		
		self.results = results
		return results
	
	def create_comparison_plots(self, save_plots: bool = True):
		"""
		Create comparison plots for all tested combinations.
		
		Args:
			save_plots: Whether to save plots to files
		"""
		if not self.results:
			print("❌ No results to plot. Run comparisons first.")
			return
		
		# Filter successful results
		successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
		
		if not successful_results:
			print("❌ No successful results to plot.")
			return
		
		# Create comprehensive comparison plots
		fig, axes = plt.subplots(2, 2, figsize=(15, 12))
		fig.suptitle('🎯 Sentiment Analysis Comparison - All Combinations', fontsize=16, fontweight='bold')
		
		# Plot 1: Accuracy Comparison
		names = list(successful_results.keys())
		accuracies = [successful_results[name]['accuracy'] for name in names]
		
		axes[0, 0].bar(range(len(names)), accuracies, color='skyblue', alpha=0.7)
		axes[0, 0].set_title('📊 Accuracy by Combination')
		axes[0, 0].set_ylabel('Accuracy')
		axes[0, 0].set_xticks(range(len(names)))
		axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
		axes[0, 0].set_ylim(0, 1)
		
		# Add accuracy values on bars
		for i, acc in enumerate(accuracies):
			axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
		
		# Plot 2: Processing Time Comparison
		times = [successful_results[name]['processing_time'] for name in names]
		
		axes[0, 1].bar(range(len(names)), times, color='lightcoral', alpha=0.7)
		axes[0, 1].set_title('⏱️ Processing Time by Combination')
		axes[0, 1].set_ylabel('Time (seconds)')
		axes[0, 1].set_xticks(range(len(names)))
		axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
		
		# Plot 3: Accuracy by Preprocessing Type
		prep_accuracies = {'basic': [], 'light': []}
		for name, result in successful_results.items():
			prep_type = result['preprocessing']
			prep_accuracies[prep_type].append(result['accuracy'])
		
		prep_means = {k: np.mean(v) if v else 0 for k, v in prep_accuracies.items()}
		axes[1, 0].bar(prep_means.keys(), prep_means.values(), color='lightgreen', alpha=0.7)
		axes[1, 0].set_title('📝 Average Accuracy by Preprocessing')
		axes[1, 0].set_ylabel('Average Accuracy')
		axes[1, 0].set_ylim(0, 1)
		
		# Plot 4: Accuracy by Model Type
		model_accuracies = {'vader': [], 'bert': []}
		for name, result in successful_results.items():
			model_type = result['model']
			model_accuracies[model_type].append(result['accuracy'])
		
		model_means = {k: np.mean(v) if v else 0 for k, v in model_accuracies.items()}
		axes[1, 1].bar(model_means.keys(), model_means.values(), color='gold', alpha=0.7)
		axes[1, 1].set_title('🤖 Average Accuracy by Model')
		axes[1, 1].set_ylabel('Average Accuracy')
		axes[1, 1].set_ylim(0, 1)
		
		plt.tight_layout()
		
		if save_plots:
			plt.savefig('sentiment_comparison_plots.png', dpi=300, bbox_inches='tight')
			print("📈 Plots saved as 'sentiment_comparison_plots.png'")
		
		plt.show()
	
	def print_detailed_results(self):
		"""Print detailed results for all combinations."""
		if not self.results:
			print("❌ No results to display. Run comparisons first.")
			return
		
		print("\n" + "="*80)
		print("📋 DETAILED SENTIMENT ANALYSIS RESULTS")
		print("="*80)
		
		# Sort results by accuracy (successful ones first)
		successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
		failed_results = {k: v for k, v in self.results.items() if 'error' in v}
		
		# Sort successful results by accuracy
		sorted_successful = sorted(successful_results.items(), 
								 key=lambda x: x[1]['accuracy'], 
								 reverse=True)
		
		print(f"\n✅ SUCCESSFUL COMBINATIONS ({len(sorted_successful)}):")
		print("-" * 50)
		
		for i, (name, result) in enumerate(sorted_successful, 1):
			print(f"\n{i}. {name.upper()}")
			print(f"   📊 Accuracy: {result['accuracy']:.3f}")
			print(f"   ⏱️  Time: {result['processing_time']:.2f}s")
			print(f"   📝 Preprocessing: {result['preprocessing']}")
			print(f"   🧠 Embedding: {result['embedding']}")
			print(f"   🤖 Model: {result['model']}")
		
		if failed_results:
			print(f"\n❌ FAILED COMBINATIONS ({len(failed_results)}):")
			print("-" * 50)
			
			for name, result in failed_results.items():
				print(f"\n• {name.upper()}")
				print(f"   Error: {result['error']}")
		
		# Best combination
		if sorted_successful:
			best_name, best_result = sorted_successful[0]
			print(f"\n🏆 BEST COMBINATION: {best_name.upper()}")
			print(f"   📊 Accuracy: {best_result['accuracy']:.3f}")
			print(f"   ⏱️  Time: {best_result['processing_time']:.2f}s")
		
		print("\n" + "="*80) 