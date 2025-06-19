import nltk
from nltk.tokenize import word_tokenize
import re
import os

class TextTokenizer:
	"""
	Enhanced tokenizer optimized for sentiment analysis.
	Handles contractions, expressive repetitions, and emotional indicators.
	"""
	
	def __init__(self, nltk_dir=None) -> None:
		"""
		Initialize the tokenizer.
		
		Args:
			nltk_dir (str): Path to NLTK data directory (optional)
		"""
		if nltk_dir:
			nltk.data.path.append(os.path.abspath(nltk_dir))
		
		# Define important contractions to preserve as single tokens
		self.important_contractions = {
			"don't", "won't", "can't", "shouldn't", "wouldn't", "couldn't", 
			"isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
			"hadn't", "didn't", "doesn't", "I'm", "you're", "he's", "she's",
			"it's", "we're", "they're", "I've", "you've", "we've", "they've",
			"I'll", "you'll", "he'll", "she'll", "we'll", "they'll"
		}
	
	def normalize_expressive_lengthening(self, text) -> str:
		"""
		Normalize expressive character repetitions while preserving emphasis.
		
		Args:
			text (str): Input text
			
		Returns:
			str: Text with normalized repetitions
		"""
		if not isinstance(text, str):
			return ""
		
		# Normalize repeated characters (3+ repetitions) to 2 repetitions
		# This preserves the expressive nature while standardizing
		# Examples: "sooooo" -> "soo", "yesss" -> "yess", "noooo" -> "noo"
		text = re.sub(r'(.)\1{2,}', r'\1\1', text)
		
		return text
	
	def preserve_contractions(self, text) -> str:
		"""
		Mark important contractions to prevent them from being split.
		
		Args:
			text (str): Input text
			
		Returns:
			str: Text with protected contractions
		"""
		if not isinstance(text, str):
			return ""
		
		# Temporarily replace important contractions with placeholders
		protected_text = text
		contraction_map = {}
		
		for i, contraction in enumerate(self.important_contractions):
			placeholder = f"__CONTRACTION_{i}__"
			# Case-insensitive replacement
			pattern = re.compile(re.escape(contraction), re.IGNORECASE)
			matches = pattern.findall(protected_text)
			
			if matches:
				# Store the original case
				contraction_map[placeholder] = matches[0]
				protected_text = pattern.sub(placeholder, protected_text)
		
		return protected_text, contraction_map
	
	def restore_contractions(self, tokens, contraction_map) -> list[str]:
		"""
		Restore protected contractions in the token list.
		
		Args:
			tokens (list): List of tokens
			contraction_map (dict): Mapping of placeholders to original contractions
			
		Returns:
			list: Tokens with restored contractions
		"""
		restored_tokens = []
		
		for token in tokens:
			if token in contraction_map:
				restored_tokens.append(contraction_map[token])
			else:
				restored_tokens.append(token)
		
		return restored_tokens
	
	def handle_punctuation_emphasis(self, text) -> str:
		"""
		Handle repeated punctuation for emotional emphasis.
		
		Args:
			text (str): Input text
			
		Returns:
			str: Text with handled punctuation emphasis
		"""
		if not isinstance(text, str):
			return ""
		
		# Preserve emotional punctuation patterns as single units
		# Examples: "!!!" -> "!!!", "???" -> "???"
		# But normalize excessive repetition: "!!!!!" -> "!!!"
		text = re.sub(r'!{2,}', '!!!', text)
		text = re.sub(r'\?{2,}', '???', text)
		
		return text
	
	def separate_emojis(self, text) -> str:
		"""
		Ensure emojis are separated from adjacent words.
		
		Args:
			text (str): Input text
			
		Returns:
			str: Text with separated emojis
		"""
		if not isinstance(text, str):
			return ""
		
		# Add spaces around emojis (basic emoji pattern)
		# This regex catches most common emojis
		emoji_pattern = r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]|[\U00002700-\U000027BF]|[\U0001f900-\U0001f9ff]'
		text = re.sub(f'({emoji_pattern})', r' \1 ', text)
		
		# Clean up multiple spaces
		text = re.sub(r'\s+', ' ', text).strip()
		
		return text
	
	def tokenize(self, text) -> list[str]:
		"""
		Enhanced tokenization preserving emotional and sentiment indicators.
		
		Args:
			text (str): The input text to tokenize
			
		Returns:
			list[str]: List of tokens optimized for sentiment analysis
		"""
		if not isinstance(text, str):
			return []
		
		# Step 1: Normalize expressive lengthening
		text = self.normalize_expressive_lengthening(text)
		
		# Step 2: Handle punctuation emphasis
		text = self.handle_punctuation_emphasis(text)
		
		# Step 3: Separate emojis
		text = self.separate_emojis(text)
		
		# Step 4: Protect important contractions
		protected_text, contraction_map = self.preserve_contractions(text)
		
		# Step 5: Tokenize
		tokens = word_tokenize(protected_text)
		
		# Step 6: Restore contractions
		tokens = self.restore_contractions(tokens, contraction_map)
		
		# Step 7: Filter out empty tokens
		tokens = [token for token in tokens if token.strip()]
		
		return tokens 