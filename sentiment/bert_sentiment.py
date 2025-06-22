"""
BERT sentiment analysis implementation.

Uses pre-trained BERT models fine-tuned for sentiment classification.
"""

from typing import List, Dict, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class BertSentiment:
	"""
	BERT sentiment analyzer using pre-trained models.
	
	Uses Hugging Face transformers for sentiment classification.
	"""
	
	def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
		"""
		Initialize BERT sentiment analyzer.
		
		Args:
			model_name: Pre-trained model name for sentiment analysis
		"""
		self.model_name = model_name
		self.classifier = None
		self._is_fitted = False
	
	def fit(self, documents: List[Union[List[str], str]]) -> 'BertSentiment':
		"""
		Load pre-trained BERT sentiment model.
		
		Args:
			documents: List of documents (not used for training, just for initialization)
			
		Returns:
			self
		"""
		if not documents:
			raise ValueError("Documents cannot be empty")
		
		# Initialize the sentiment analysis pipeline
		self.classifier = pipeline(
			"sentiment-analysis",
			model=self.model_name,
			tokenizer=self.model_name,
			device=0 if torch.cuda.is_available() else -1  # Use GPU if available
		)
		
		self._is_fitted = True
		return self
	
	def predict_sentiment(self, documents: List[Union[List[str], str]]) -> Dict[str, np.ndarray]:
		"""
		Predict sentiment for documents.
		
		Args:
			documents: List of documents (lists of tokens or raw strings)
			
		Returns:
			Dict containing sentiment scores and classifications
		"""
		if not self._is_fitted:
			raise ValueError("Call fit() first")
		
		if not documents:
			return {
				'scores': np.array([]),
				'classification': np.array([])
			}
		
		# Convert documents to text if they are tokenized
		texts = []
		for doc in documents:
			if isinstance(doc, list):
				text = " ".join(doc)
			else:
				text = doc
			texts.append(text)
		
		# Predict sentiment for all texts
		predictions = self.classifier(texts)
		
		# Extract results
		scores = []
		classifications = []
		
		for pred in predictions:
			scores.append(pred['score'])
			
			# Normalize label names
			label = pred['label'].upper()
			if label in ['POSITIVE', 'POS']:
				classification = 'positive'
			elif label in ['NEGATIVE', 'NEG']:
				classification = 'negative'
			else:
				classification = 'neutral'
			
			classifications.append(classification)
		
		return {
			'scores': np.array(scores, dtype=np.float32),
			'classification': np.array(classifications)
		}
	
	def fit_predict(self, documents: List[Union[List[str], str]]) -> Dict[str, np.ndarray]:
		"""
		Fit and predict in one step.
		
		Args:
			documents: List of documents
			
		Returns:
			Dict containing sentiment predictions
		"""
		return self.fit(documents).predict_sentiment(documents)