import re
import os
from enum import Enum
from abc import ABC, abstractmethod
from typing import List
from bs4 import BeautifulSoup

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Configure NLTK data path
nltk_dir = 'nltk_data'
nltk.data.path.append(os.path.abspath(nltk_dir))

# Import emoji for emoji handling
try:
	import emoji
except ImportError:
	print("Warning: emoji package not installed. Install with: pip install emoji")
	emoji = None

class CleaningLevel(Enum):
	"""Enumeration for different cleaning levels."""
	LIGHT = "light"
	MEDIUM = "medium"
	HARD = "hard"
	CUSTOM = "custom"

class BaseProcessor(ABC):
	"""Abstract base class for text processors."""
	
	@abstractmethod
	def process(self, text: str) -> List[str]:
		"""Process text and return list of tokens."""
		pass

class LightProcessor(BaseProcessor):
	"""
	Light cleaning - Preserve sentiment indicators
	
	Philosophy: Keep as much original information as possible
	- Preserve punctuation for emphasis (!!!, ???)
	- Keep emojis for emotional context
	- Keep contractions for natural speech patterns
	- Keep hashtags for topic context
	- NO lemmatization to preserve meaning nuances
	
	Best for: Sentiment analysis where emphasis and emotion matter
	"""
	
	def __init__(self):
		"""Initialize light processor with minimal components."""
		# Only initialize what we need
		try:
			self.stop_words = set(stopwords.words('english'))
		except:
			print("Warning: NLTK stopwords not available")
			self.stop_words = set()
	
	def process(self, text: str) -> List[str]:
		"""
		Apply light cleaning to preserve sentiment indicators.
		
		Args:
			text: Input text to process
			
		Returns:
			List of lightly processed tokens
		"""
		if not isinstance(text, str):
			return []
		
		result = text
		
		# Step 1: Remove only essential noise
		result = BeautifulSoup(result, "html.parser").get_text()  # Remove HTML
		
		# Step 2: Convert to lowercase (but preserve emphasis first)
		# Preserve multiple punctuation as emphasis markers
		result = re.sub(r'!{2,}', ' VERY_EXCITED ', result)
		result = re.sub(r'\?{2,}', ' VERY_QUESTIONING ', result)
		result = re.sub(r'\.{3,}', ' TRAILING_THOUGHT ', result)
		
		# Preserve CAPS words as emphasis
		result = re.sub(r'\b[A-Z]{2,}\b', lambda m: f'EMPHASIS_{m.group().lower()}', result)
		
		result = result.lower()
		
		# Step 3: Remove only URLs and mentions (keep hashtags!)
		result = re.sub(r"https?://\S+", "", result)  # Remove URLs
		result = re.sub(r"@\w+", "", result)         # Remove mentions
		
		# Step 4: Keep hashtags, punctuation, emojis, numbers!
		# Only normalize whitespace
		result = re.sub(r"\s+", " ", result).strip()
		
		# Step 5: Tokenize
		try:
			tokens = word_tokenize(result)
		except:
			# Fallback tokenization if NLTK not available
			tokens = result.split()
		
		# Step 6: Very light stopword removal (keep negations and short words)
		negations = {"not", "no", "never", "n't", "dont", "wont", "cant"}
		tokens = [word for word in tokens 
		         if word.lower() not in self.stop_words 
		         or word.lower() in negations 
		         or len(word) <= 2]  # Keep short words like "so", "oh"
		
		# Step 7: NO lemmatization - preserve original word forms
		return tokens

class MediumProcessor(BaseProcessor):
	"""
	Medium cleaning - Balanced approach
	
	Philosophy: Clean but preserve sentiment-critical elements
	- Remove most punctuation but preserve key patterns
	- Remove emojis but keep their sentiment impact
	- Selective lemmatization (preserve adjectives!)
	- Expand contractions for consistency
	
	Best for: General sentiment analysis with some normalization
	"""
	
	def __init__(self):
		"""Initialize medium processor."""
		try:
			self.stop_words = set(stopwords.words('english'))
			self.lemmatizer = WordNetLemmatizer()
		except:
			print("Warning: NLTK components not fully available")
			self.stop_words = set()
			self.lemmatizer = None
	
	def get_wordnet_pos(self, treebank_tag: str) -> str:
		"""Convert POS tag from TreeBank to WordNet format."""
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			return wordnet.NOUN
	
	def expand_contractions(self, text: str) -> str:
		"""Expand common contractions."""
		contractions = {
			"won't": "will not", "can't": "cannot", "n't": " not",
			"'re": " are", "'ve": " have", "'ll": " will", 
			"'d": " would", "'m": " am", "'s": " is"
		}
		
		for contraction, expansion in contractions.items():
			text = text.replace(contraction, expansion)
		return text
	
	def process(self, text: str) -> List[str]:
		"""
		Apply medium cleaning with selective processing.
		
		Args:
			text: Input text to process
			
		Returns:
			List of moderately processed tokens
		"""
		if not isinstance(text, str):
			return []
		
		result = text
		
		# Step 1: Remove HTML
		result = BeautifulSoup(result, "html.parser").get_text()
		
		# Step 2: Expand contractions before other processing
		result = self.expand_contractions(result)
		
		# Step 3: Convert to lowercase
		result = result.lower()
		
		# Step 4: Remove URLs, mentions, and hashtags
		result = re.sub(r"https?://\S+", "", result)
		result = re.sub(r"@\w+", "", result)
		result = re.sub(r"#\w+", "", result)
		
		# Step 5: Remove emojis but preserve their sentiment meaning
		if emoji:
			# Convert some emojis to sentiment words before removal
			result = result.replace("😍", " love ").replace("😢", " sad ")
			result = result.replace("😊", " happy ").replace("😡", " angry ")
			result = emoji.replace_emoji(result, replace='')
		
		# Step 6: Remove punctuation and digits
		result = re.sub(r"[^\w\s]", "", result)
		result = re.sub(r"\d+", "", result)
		
		# Step 7: Normalize whitespace
		result = re.sub(r"\s+", " ", result).strip()
		
		# Step 8: Tokenize
		try:
			tokens = word_tokenize(result)
		except:
			tokens = result.split()
		
		# Step 9: Remove stopwords but keep negations
		negations = {"not", "no", "never"}
		tokens = [word for word in tokens 
		         if word.lower() not in self.stop_words 
		         or word.lower() in negations]
		
		# Step 10: Selective lemmatization - DON'T lemmatize adjectives!
		if self.lemmatizer:
			try:
				pos_tags = nltk.pos_tag(tokens)
				lemmatized = []
				
				for token, pos in pos_tags:
					# Preserve adjectives to keep sentiment intensity
					if pos.startswith('JJ'):  # Adjectives (better, worse, amazing, etc.)
						lemmatized.append(token)
					else:
						lemmatized.append(
							self.lemmatizer.lemmatize(token, self.get_wordnet_pos(pos))
						)
				
				return lemmatized
			except:
				pass
		
		return tokens

class HardProcessor(BaseProcessor):
	"""
	Hard cleaning - Aggressive normalization
	
	Philosophy: Maximum normalization for clean, standardized text
	- Remove ALL non-essential elements
	- Full lemmatization
	- Strict filtering
	- Aggressive length filtering
	
	Best for: Topic modeling, keyword extraction, formal analysis
	"""
	
	def __init__(self):
		"""Initialize hard processor."""
		try:
			self.stop_words = set(stopwords.words('english'))
			self.lemmatizer = WordNetLemmatizer()
		except:
			print("Warning: NLTK components not fully available")
			self.stop_words = set()
			self.lemmatizer = None
	
	def get_wordnet_pos(self, treebank_tag: str) -> str:
		"""Convert POS tag from TreeBank to WordNet format."""
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			return wordnet.NOUN
	
	def process(self, text: str) -> List[str]:
		"""
		Apply aggressive cleaning for maximum normalization.
		
		Args:
			text: Input text to process
			
		Returns:
			List of heavily processed tokens
		"""
		if not isinstance(text, str):
			return []
		
		result = text
		
		# Step 1: Remove HTML
		result = BeautifulSoup(result, "html.parser").get_text()
		
		# Step 2: Convert to lowercase
		result = result.lower()
		
		# Step 3: Remove ALL URLs, mentions, hashtags
		result = re.sub(r"https?://\S+", "", result)
		result = re.sub(r"www\.\S+", "", result)  # Also remove www links
		result = re.sub(r"@\w+", "", result)
		result = re.sub(r"#\w+", "", result)
		
		# Step 4: Remove ALL emojis
		if emoji:
			result = emoji.replace_emoji(result, replace='')
		
		# Step 5: Remove ALL punctuation and digits
		result = re.sub(r"[^\w\s]", "", result)
		result = re.sub(r"\d+", "", result)
		
		# Step 6: Normalize whitespace
		result = re.sub(r"\s+", " ", result).strip()
		
		# Step 7: Tokenize
		try:
			tokens = word_tokenize(result)
		except:
			tokens = result.split()
		
		# Step 8: Aggressive filtering
		# Filter by length (keep only words >= 3 characters)
		tokens = [token for token in tokens if len(token) >= 3]
		
		# Step 9: Remove stopwords (but keep negations for sentiment)
		negations = {"not", "never"}
		tokens = [word for word in tokens 
		         if word.lower() not in self.stop_words 
		         or word.lower() in negations]
		
		# Step 10: Full lemmatization
		if self.lemmatizer:
			try:
				pos_tags = nltk.pos_tag(tokens)
				lemmatized = []
				
				for token, pos in pos_tags:
					lemmatized.append(
						self.lemmatizer.lemmatize(token, self.get_wordnet_pos(pos))
					)
				
				return lemmatized
			except:
				pass
		
		return tokens

class ProcessorFactory:
	"""Factory to create appropriate processor based on cleaning level."""
	
	@staticmethod
	def create_processor(level: CleaningLevel) -> BaseProcessor:
		"""
		Create processor based on cleaning level.
		
		Args:
			level: CleaningLevel enum value
			
		Returns:
			Appropriate processor instance
		"""
		if level == CleaningLevel.LIGHT:
			return LightProcessor()
		elif level == CleaningLevel.MEDIUM:
			return MediumProcessor()
		elif level == CleaningLevel.HARD:
			return HardProcessor()
		else:
			raise ValueError(f"Unsupported cleaning level: {level}")
	
	@staticmethod
	def process_text(text: str, level: CleaningLevel) -> List[str]:
		"""
		Quick processing with specified level.
		
		Args:
			text: Input text to process
			level: Cleaning level to apply
			
		Returns:
			List of processed tokens
		"""
		processor = ProcessorFactory.create_processor(level)
		return processor.process(text)
	
	@staticmethod
	def compare_levels(text: str) -> dict:
		"""
		Compare results across all cleaning levels.
		
		Args:
			text: Input text to process
			
		Returns:
			Dictionary with results for each level
		"""
		results = {}
		
		for level in [CleaningLevel.LIGHT, CleaningLevel.MEDIUM, CleaningLevel.HARD]:
			try:
				results[level.value] = ProcessorFactory.process_text(text, level)
			except Exception as e:
				results[level.value] = f"Error: {str(e)}"
		
		return results

# Convenience functions for easy usage
def light_clean(text: str) -> List[str]:
	"""Apply light cleaning to text."""
	return ProcessorFactory.process_text(text, CleaningLevel.LIGHT)

def medium_clean(text: str) -> List[str]:
	"""Apply medium cleaning to text."""
	return ProcessorFactory.process_text(text, CleaningLevel.MEDIUM)

def hard_clean(text: str) -> List[str]:
	"""Apply hard cleaning to text."""
	return ProcessorFactory.process_text(text, CleaningLevel.HARD)

def compare_cleaning(text: str) -> dict:
	"""Compare all cleaning levels on the same text."""
	return ProcessorFactory.compare_levels(text)

# Example usage
if __name__ == "__main__":
	# Test text with various challenges
	test_text = "I'm ABSOLUTELY loving the new ChatGPT updates!!! It's SO much better than before... 😍 #AI #ChatGPT"
	
	print("=== FLEXIBLE PREPROCESSING COMPARISON ===\n")
	print(f"Original: {test_text}\n")
	
	# Compare all levels
	results = compare_cleaning(test_text)
	
	for level, tokens in results.items():
		print(f"{level.upper()}: {tokens}")
	
	print("\n" + "="*50)
	print("Key differences:")
	print("- LIGHT: Preserves emphasis, emojis, hashtags")
	print("- MEDIUM: Balanced, selective lemmatization") 
	print("- HARD: Aggressive normalization") 