"""
Simple Word2Vec implementation for text vectorization.

Converts tokens to numerical vectors using Word2Vec embeddings,
then aggregates word vectors to document vectors.
"""

from typing import List, Optional
import numpy as np
from gensim.models import Word2Vec as GensimWord2Vec


class Word2Vec:
	"""
	Simple Word2Vec vectorizer.
	
	Converts documents (lists of tokens) into numerical vectors
	by training Word2Vec embeddings and aggregating word vectors.
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
		self.model_: Optional[GensimWord2Vec] = None
		self._is_fitted = False
	
	def fit(self, documents: List[List[str]]) -> 'Word2Vec':
		"""
		Train Word2Vec model on documents.
		
		Args:
			documents: List of documents, where each document is a list of tokens.
			
		Returns:
			self
		"""
		if not documents:
			raise ValueError("Documents cannot be empty")
		
		# Train Word2Vec model
		self.model_ = GensimWord2Vec(
			sentences=documents,
			vector_size=self.vector_size,
			min_count=self.min_count,
			workers=1,  # Single thread for consistency
			seed=42     # Reproducible results
		)
		
		self._is_fitted = True
		return self
	
	def transform(self, documents: List[List[str]]) -> np.ndarray:
		"""
		Convert documents to Word2Vec vectors.
		
		Each document vector is the average of its word vectors.
		
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
			
			# Get vectors for words that exist in vocabulary
			word_vectors = []
			for word in doc:
				if word in self.model_.wv:
					word_vectors.append(self.model_.wv[word])
			
			# Average word vectors to get document vector
			if word_vectors:
				vectors[doc_idx] = np.mean(word_vectors, axis=0)
			# else: keep zero vector for documents with no known words
		
		return vectors
	
	def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
		"""
		Fit and transform in one step.
		
		Args:
			documents: List of documents (lists of tokens).
			
		Returns:
			np.ndarray: Document Word2Vec vectors.
		"""
		return self.fit(documents).transform(documents) 