"""
Simple FastText implementation for text vectorization.

Converts tokens to numerical vectors using FastText embeddings,
then aggregates word vectors to document vectors.
"""

from typing import List, Optional
import numpy as np
from gensim.models import FastText as GensimFastText


class FastText:
	"""
	Simple FastText vectorizer.
	
	Converts documents (lists of tokens) into numerical vectors
	by training FastText embeddings and aggregating word vectors.
	FastText can handle out-of-vocabulary words using subword information.
	"""
	
	def __init__(self, vector_size: int = 100, min_count: int = 1):
		"""
		Initialize the vectorizer.
		
		Args:
			vector_size: Size of word vectors (default: 100)
			min_count: Minimum word frequency to include (default: 1)
		"""
		self.vector_size = vector_size
		self.min_count = min_count
		self.model_: Optional[GensimFastText] = None
		self._is_fitted = False
	
	def fit(self, documents: List[List[str]]) -> 'FastText':
		"""
		Train FastText model on documents.
		
		Args:
			documents: List of documents, where each document is a list of tokens.
			
		Returns:
			self
		"""
		if not documents:
			raise ValueError("Documents cannot be empty")
		
		# Train FastText model
		self.model_ = GensimFastText(
			sentences=documents,
			vector_size=self.vector_size,
			min_count=self.min_count,
			workers=6,  # Single thread for consistency
			seed=42,    # Reproducible results
			min_n=3,    # Minimum subword length
			max_n=6     # Maximum subword length
		)
		
		self._is_fitted = True
		return self
	
	def transform(self, documents: List[List[str]]) -> np.ndarray:
		"""
		Convert documents to FastText vectors.
		
		Each document vector is the average of its word vectors.
		FastText can generate vectors even for unknown words.
		
		Args:
			documents: List of documents (lists of tokens).
			
		Returns:
			np.ndarray: Matrix where each row is a document vector.
		"""
		if not self._is_fitted:
			raise ValueError("Call fit() first")
		
		if not documents:
			return np.zeros((0, self.vector_size))
		
		# Initialize result matrix
		n_docs = len(documents)
		vectors = np.zeros((n_docs, self.vector_size), dtype=np.float32)
		
		# Convert each document to vector (average of word vectors)
		for doc_idx, doc in enumerate(documents):
			if not doc:  # Skip empty documents
				continue
			
			# Get vectors for all words (FastText handles OOV words)
			word_vectors = []
			for word in doc:
				try:
					# FastText can generate vectors even for unseen words
					word_vectors.append(self.model_.wv[word])
				except KeyError:
					# Should rarely happen with FastText, but handle gracefully
					continue
			
			# Average word vectors to get document vector
			if word_vectors:
				vectors[doc_idx] = np.mean(word_vectors, axis=0)
			# else: keep zero vector for documents with no processable words
		
		return vectors
	
	def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
		"""
		Fit and transform in one step.
		
		Args:
			documents: List of documents (lists of tokens).
			
		Returns:
			np.ndarray: Document FastText vectors.
		"""
		return self.fit(documents).transform(documents) 