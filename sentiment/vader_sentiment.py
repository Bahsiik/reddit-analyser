"""
VADER sentiment analysis implementation.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based 
sentiment analysis tool specifically attuned to sentiments expressed in social media.
"""

from typing import List, Dict, Union
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderSentiment:
	"""
	VADER sentiment analyzer.
	
	Uses lexicon-based approach for sentiment classification.
	Returns sentiment scores and classifications.
	"""
	
	def __init__(self):
		"""Initialize VADER sentiment analyzer."""
		self.analyzer = SentimentIntensityAnalyzer()
		self._is_fitted = False
	
	def fit(self, documents: List[Union[List[str], str]]) -> 'VaderSentiment':
		"""
		VADER doesn't require training (lexicon-based).
		This method is here for consistency with other models.
		
		Args:
			documents: List of documents (not used for VADER)
			
		Returns:
			self
		"""
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
				'compound': np.array([]),
				'positive': np.array([]),
				'negative': np.array([]),
				'neutral': np.array([]),
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
		
		# Analyze sentiment for each text
		results = {
			'compound': [],
			'positive': [],
			'negative': [],
			'neutral': [],
			'classification': []
		}
		
		for text in texts:
			scores = self.analyzer.polarity_scores(text)
			
			results['compound'].append(scores['compound'])
			results['positive'].append(scores['pos'])
			results['negative'].append(scores['neg'])
			results['neutral'].append(scores['neu'])
			
			# Classify based on compound score
			if scores['compound'] >= 0.05:
				classification = 'positive'
			elif scores['compound'] <= -0.05:
				classification = 'negative'
			else:
				classification = 'neutral'
			
			results['classification'].append(classification)
		
		# Convert to numpy arrays
		for key in results:
			if key == 'classification':
				results[key] = np.array(results[key])
			else:
				results[key] = np.array(results[key], dtype=np.float32)
		
		return results
	
	def fit_predict(self, documents: List[Union[List[str], str]]) -> Dict[str, np.ndarray]:
		"""
		Fit and predict in one step.
		
		Args:
			documents: List of documents
			
		Returns:
			Dict containing sentiment predictions
		"""
		return self.fit(documents).predict_sentiment(documents)