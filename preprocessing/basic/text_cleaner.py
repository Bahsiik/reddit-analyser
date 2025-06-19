import re
from bs4 import BeautifulSoup

class TextCleaner:
	"""
	Class for cleaning and preprocessing text data with multiple processing steps.
	"""
	
	def __init__(self) -> None:
		pass
	
	def remove_html(self, text) -> str:
		"""Remove HTML tags from text."""
		if not isinstance(text, str):
			return ""
		return BeautifulSoup(text, "html.parser").get_text()
	
	def convert_to_lowercase(self, text) -> str:
		"""Convert text to lowercase."""
		if not isinstance(text, str):
			return ""
		return text.lower()
	
	def remove_urls(self, text) -> str:
		"""Remove URLs from text."""
		if not isinstance(text, str):
			return ""
		return re.sub(r"http\S+", "", text)
	
	def remove_mentions_hashtags(self, text) -> str:
		"""Remove mentions (@username) and hashtags (#hashtag) from text."""
		if not isinstance(text, str):
			return ""
		return re.sub(r"@\w+|#\w+", "", text)
	
	def remove_punctuation(self, text) -> str:
		"""Remove punctuation from text."""
		if not isinstance(text, str):
			return ""
		return re.sub(r"[^\w\s]", "", text)
	
	def remove_digits(self, text) -> str:
		"""Remove digits from text."""
		if not isinstance(text, str):
			return ""
		return re.sub(r"\d+", "", text)
	
	def normalize_whitespace(self, text) -> str:
		"""Normalize whitespace in text."""
		if not isinstance(text, str):
			return ""
		return re.sub(r"\s+", " ", text).strip()
	
	def clean_text(self, text) -> str:
		"""
		Apply all cleaning steps to the text.
		
		Args:
			text (str): The text to clean
			
		Returns:
			str: The cleaned text
		"""
		if not isinstance(text, str):
			return ""
		
		# Apply all cleaning steps in sequence
		text = self.remove_html(text)
		text = self.convert_to_lowercase(text)
		text = self.remove_urls(text)
		text = self.remove_mentions_hashtags(text)
		text = self.remove_punctuation(text)
		text = self.remove_digits(text)
		text = self.normalize_whitespace(text)
		
		return text

cleaner = TextCleaner()

def clean_text(text) -> str:
	return cleaner.clean_text(text)
