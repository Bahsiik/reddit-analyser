"""
Simple TF-IDF implementation for text vectorization.

Converts tokens to numerical vectors based on Term Frequency - Inverse Document Frequency.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import Counter
import math


class TfIdf:
	"""
	Simple TF-IDF vectorizer.
	
	Converts documents (lists of tokens) into numerical vectors
	where each dimension represents a TF-IDF score.
	"""
	
	def __init__(self):
		"""Initialize the vectorizer."""
		self.vocabulary_: Optional[Dict[str, int]] = None
		self.idf_: Optional[np.ndarray] = None
		self._is_fitted = False
	
	def fit(self, documents: List[List[str]]) -> 'TfIdf':
		"""
		Build vocabulary and compute IDF scores from documents.
		
		Args:
			documents: List of documents, where each document is a list of tokens.
			
		Returns:
			self
		"""
		if not documents:
			raise ValueError("Documents cannot be empty")
		
		# Collect all unique words
		all_words = set()
		for doc in documents:
			all_words.update(doc)
		
		# Build vocabulary mapping (sorted for consistency)
		words = sorted(all_words)
		self.vocabulary_ = {word: idx for idx, word in enumerate(words)}
		
		# Compute IDF scores
		n_docs = len(documents)
		idf_scores = np.zeros(len(words))
		
		# Count document frequency for each word
		for word_idx, word in enumerate(words):
			doc_freq = sum(1 for doc in documents if word in doc)
			# IDF = log(total_docs / doc_freq)
			idf_scores[word_idx] = math.log(n_docs / doc_freq)
		
		self.idf_ = idf_scores
		self._is_fitted = True
		
		return self
	
	def transform(self, documents: List[List[str]]) -> np.ndarray:
		"""
		Convert documents to TF-IDF vectors.
		
		Args:
			documents: List of documents (lists of tokens).
			
		Returns:
			np.ndarray: Matrix where each row is a document TF-IDF vector.
		"""
		if not self._is_fitted:
			raise ValueError("Call fit() first")
		
		if not documents:
			return np.zeros((0, len(self.vocabulary_)))
		
		# Initialize result matrix
		n_docs = len(documents)
		n_features = len(self.vocabulary_)
		vectors = np.zeros((n_docs, n_features), dtype=np.float64)
		
		# Compute TF-IDF for each document
		for doc_idx, doc in enumerate(documents):
			if not doc:  # Skip empty documents
				continue
				
			# Compute TF (Term Frequency)
			word_counts = Counter(doc)
			doc_length = len(doc)
			
			for word, count in word_counts.items():
				if word in self.vocabulary_:
					word_idx = self.vocabulary_[word]
					# TF = word_count / doc_length
					tf = count / doc_length
					# TF-IDF = TF * IDF
					vectors[doc_idx, word_idx] = tf * self.idf_[word_idx]
		
		return vectors
	
	def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
		"""
		Fit and transform in one step.
		
		Args:
			documents: List of documents (lists of tokens).
			
		Returns:
			np.ndarray: Document TF-IDF vectors.
		"""
		return self.fit(documents).transform(documents) 