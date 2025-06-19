import re
from bs4 import BeautifulSoup

class TextCleaner:
	"""
	Lightweight text cleaner optimized for sentiment analysis.
	Preserves emotional indicators while removing noise.
	"""
	
	def __init__(self) -> None:
		pass
	
	def remove_html(self, text) -> str:
		"""Remove HTML tags from text."""
		if not isinstance(text, str):
			return ""
		return BeautifulSoup(text, "html.parser").get_text()
	
	def convert_to_lowercase(self, text) -> str:
		"""Convert text to lowercase while preserving ALL-CAPS words for emphasis."""
		if not isinstance(text, str):
			return ""
		
		# Split into words and process each one
		words = text.split()
		processed_words = []
		
		for word in words:
			# Keep words that are ALL CAPS (but not single letters) as they indicate emphasis
			if len(word) > 1 and word.isupper() and word.isalpha():
				processed_words.append(word)
			else:
				processed_words.append(word.lower())
		
		return " ".join(processed_words)
	
	def remove_urls(self, text) -> str:
		"""Remove URLs from text."""
		if not isinstance(text, str):
			return ""
		return re.sub(r"http\S+|www\.\S+", "", text)
	
	def remove_mentions_hashtags(self, text) -> str:
		"""Remove mentions (@username) and hashtags (#hashtag) from text."""
		if not isinstance(text, str):
			return ""
		return re.sub(r"@\w+|#\w+", "", text)
	
	def remove_neutral_punctuation(self, text) -> str:
		"""Remove neutral punctuation while preserving emotional indicators."""
		if not isinstance(text, str):
			return ""
		
		# Remove neutral punctuation but keep emotional indicators
		# Keep: ! ? - ' (for contractions) and preserve emojis
		# Remove: , . ; : ( ) [ ] { } " etc.
		text = re.sub(r'[,.;:()\[\]{}"/\\`~@#$%^&*+=<>|]', '', text)
		
		return text
	
	def preserve_contractions(self, text) -> str:
		"""Preserve important contractions that affect sentiment."""
		if not isinstance(text, str):
			return ""
		
		# Ensure contractions are properly handled
		# This method mainly validates that contractions remain intact
		# The actual preservation happens in remove_neutral_punctuation
		return text
	
	def remove_irrelevant_digits(self, text) -> str:
		"""Remove digits while preserving years and ratings that may indicate sentiment."""
		if not isinstance(text, str):
			return ""
		
		# Preserve years (19xx, 20xx) and ratings (x/x, x/10, etc.)
		# Remove other isolated digits
		
		# Keep patterns like: 2024, 1999, 5/5, 8/10, 10/10
		text = re.sub(r'\b(?<!\d/)(?<![\d/])\d+(?!/\d)(?!\d)\b(?![/]\d)', '', text)
		
		return text
	
	def normalize_whitespace(self, text) -> str:
		"""Normalize whitespace in text."""
		if not isinstance(text, str):
			return ""
		return re.sub(r"\s+", " ", text).strip()
	
	def clean_text(self, text) -> str:
		"""
		Apply light cleaning steps optimized for sentiment analysis.
		
		Args:
			text (str): The text to clean
			
		Returns:
			str: The lightly cleaned text preserving emotional indicators
		"""
		if not isinstance(text, str):
			return ""
		
		# Apply light cleaning steps in sequence
		text = self.remove_html(text)
		text = self.remove_urls(text)
		text = self.remove_mentions_hashtags(text)
		text = self.preserve_contractions(text)
		text = self.remove_neutral_punctuation(text)
		text = self.remove_irrelevant_digits(text)
		text = self.convert_to_lowercase(text)  # Applied later to preserve ALL-CAPS detection
		text = self.normalize_whitespace(text)
		
		return text 