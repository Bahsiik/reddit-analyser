"""
Simple BERT implementation for text vectorization.

Converts text to numerical vectors using pre-trained BERT embeddings.
"""

from typing import List, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class Bert:
	"""
	Simple BERT vectorizer.
	
	Converts documents into numerical vectors using pre-trained BERT embeddings.
	Can handle both raw text and pre-tokenized text.
	"""
	
	def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
		"""
		Initialize the vectorizer.
		
		Args:
			model_name: BERT model name (default: "bert-base-uncased")
			max_length: Maximum sequence length (default: 512)
		"""
		self.model_name = model_name
		self.max_length = max_length
		self.tokenizer = None
		self.model = None
		self._is_fitted = False
	
	def fit(self, documents: List[Union[List[str], str]]) -> 'Bert':
		"""
		Load BERT model and tokenizer.
		
		Args:
			documents: List of documents (can be lists of tokens or raw strings).
			
		Returns:
			self
		"""
		if not documents:
			raise ValueError("Documents cannot be empty")
		
		# Load tokenizer and model
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		self.model = AutoModel.from_pretrained(self.model_name)
		
		# Set model to evaluation mode
		self.model.eval()
		
		self._is_fitted = True
		return self
	
	def transform(self, documents: List[Union[List[str], str]]) -> np.ndarray:
		"""
		Convert documents to BERT vectors.
		
		Args:
			documents: List of documents (lists of tokens or raw strings).
			
		Returns:
			np.ndarray: Matrix where each row is a document BERT vector.
		"""
		if not self._is_fitted:
			raise ValueError("Call fit() first")
		
		if not documents:
			return np.zeros((0, 768))  # BERT-base has 768 dimensions
		
		# Convert documents to text if they are tokenized
		texts = []
		for doc in documents:
			if isinstance(doc, list):
				# Join tokens back to text
				text = " ".join(doc)
			else:
				# Already text
				text = doc
			texts.append(text)
		
		# Get BERT embeddings
		vectors = []
		
		with torch.no_grad():
			for text in texts:
				# Tokenize and encode
				inputs = self.tokenizer(
					text,
					max_length=self.max_length,
					padding=True,
					truncation=True,
					return_tensors="pt"
				)
				
				# Get BERT outputs
				outputs = self.model(**inputs)
				
				# Use [CLS] token embedding as document representation
				cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
				vectors.append(cls_embedding)
		
		return np.array(vectors, dtype=np.float32)
	
	def fit_transform(self, documents: List[Union[List[str], str]]) -> np.ndarray:
		"""
		Fit and transform in one step.
		
		Args:
			documents: List of documents (lists of tokens or raw strings).
			
		Returns:
			np.ndarray: Document BERT vectors.
		"""
		return self.fit(documents).transform(documents) 