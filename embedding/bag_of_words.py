"""
Simple Bag of Words implementation for text vectorization.

Converts tokens to numerical vectors based on word frequency counts.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import Counter


class BagOfWords:
	"""
	Simple Bag of Words vectorizer.
	
	Converts documents (lists of tokens) into numerical vectors
	where each dimension represents a word frequency.
	"""
	
	def __init__(self):
		"""Initialize the vectorizer."""
		self.vocabulary_: Optional[Dict[str, int]] = None
		self._is_fitted = False
	
	def fit(self, documents: List[List[str]]) -> 'BagOfWords':
		"""
		Build vocabulary from documents.
		
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
		self._is_fitted = True
		
		return self
	
	def transform(self, documents: List[List[str]]) -> np.ndarray:
		"""
		Convert documents to vectors.
		
		Args:
			documents: List of documents (lists of tokens).
			
		Returns:
			np.ndarray: Matrix where each row is a document vector.
		"""
		if not self._is_fitted:
			raise ValueError("Call fit() first")
		
		if not documents:
			return np.zeros((0, len(self.vocabulary_)))
		
		# Initialize result matrix
		n_docs = len(documents)
		n_features = len(self.vocabulary_)
		vectors = np.zeros((n_docs, n_features), dtype=np.int32)
		
		# Count word frequencies for each document
		for doc_idx, doc in enumerate(documents):
			word_counts = Counter(doc)
			for word, count in word_counts.items():
				if word in self.vocabulary_:
					vectors[doc_idx, self.vocabulary_[word]] = count
		
		return vectors
	
	def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
		"""
		Fit and transform in one step.
		
		Args:
			documents: List of documents (lists of tokens).
			
		Returns:
			np.ndarray: Document vectors.
		"""
		return self.fit(documents).transform(documents) 