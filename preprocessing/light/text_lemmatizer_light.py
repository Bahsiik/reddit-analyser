import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import os

class TextLemmatizer:
	"""
	Lightweight lemmatizer optimized for sentiment analysis.
	Preserves emotional intensity and sentiment nuances while normalizing text.
	"""
	
	def __init__(self, nltk_dir=None) -> None:
		"""
		Initialize the lemmatizer.
		
		Args:
			nltk_dir (str): Path to NLTK data directory (optional)
		"""
		if nltk_dir:
			nltk.data.path.append(os.path.abspath(nltk_dir))
		
		self.lemmatizer = WordNetLemmatizer()
		
		# Words to preserve in their original form due to emotional significance
		self.emotional_preserves = {
			# Comparative/superlative forms with different emotional intensity
			"better", "worse", "best", "worst", "greater", "lesser",
			
			# High-intensity emotional words
			"amazing", "awesome", "terrible", "horrible", "fantastic", 
			"wonderful", "awful", "brilliant", "outstanding", "excellent",
			"disgusting", "revolting", "magnificent", "marvelous",
			
			# Intensity modifiers that shouldn't be lemmatized
			"extremely", "incredibly", "absolutely", "totally", "completely",
			"utterly", "perfectly", "entirely", "purely", "truly",
			
			# Emotional states that lose meaning when lemmatized
			"loving", "hating", "adoring", "despising", "cherishing",
			
			# Expressive forms (from tokenizer normalization)
			"soo", "yess", "noo", "woo", "hmm", "ohh", "ahh"
		}
		
		# Contractions that need special handling
		self.contraction_mapping = {
			"can't": ["can", "'t"],
			"won't": ["will", "not"],
			"don't": ["do", "not"],
			"doesn't": ["does", "not"],
			"didn't": ["did", "not"],
			"shouldn't": ["should", "not"],
			"wouldn't": ["would", "not"],
			"couldn't": ["could", "not"],
			"isn't": ["is", "not"],
			"aren't": ["are", "not"],
			"wasn't": ["was", "not"],
			"weren't": ["were", "not"],
			"haven't": ["have", "not"],
			"hasn't": ["has", "not"],
			"hadn't": ["had", "not"]
		}
	
	def get_wordnet_pos(self, treebank_tag) -> str:
		"""
		Convert POS tag from Treebank to WordNet format for better lemmatization.
		"""
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			return wordnet.NOUN  # fallback
	
	def handle_contractions(self, tokens) -> list[str]:
		"""
		Handle contractions by expanding them for better lemmatization.
		
		Args:
			tokens (list): List of tokens
			
		Returns:
			list: Tokens with handled contractions
		"""
		processed_tokens = []
		
		for token in tokens:
			token_lower = token.lower()
			
			if token_lower in self.contraction_mapping:
				# Expand the contraction
				expanded = self.contraction_mapping[token_lower]
				processed_tokens.extend(expanded)
			else:
				processed_tokens.append(token)
		
		return processed_tokens
	
	def normalize_expressive_forms(self, token) -> str:
		"""
		Normalize expressive forms for better lemmatization.
		
		Args:
			token (str): Input token
			
		Returns:
			str: Normalized token
		"""
		# Handle expressive lengthening that survived tokenization
		# Examples: "loove" -> "love", "goood" -> "good"
		
		# Don't normalize if it's in our preserve list
		if token.lower() in self.emotional_preserves:
			return token
		
		# Only normalize if we have repeated characters (from tokenizer: max 2 repetitions)
		# and the word isn't a real word as-is
		if len(set(token)) < len(token) and len(token) > 3:
			# Try to find the root by removing extra characters
			import string
			normalized = ""
			prev_char = ""
			for char in token:
				if char != prev_char or len(normalized) == 0:
					normalized += char
				prev_char = char
			
			# Only return normalized version if it looks like a real word
			if len(normalized) >= 3:
				return normalized
		
		return token
	
	def should_preserve_token(self, token, pos_tag) -> bool:
		"""
		Determine if a token should be preserved without lemmatization.
		
		Args:
			token (str): The token to check
			pos_tag (str): POS tag of the token
			
		Returns:
			bool: Whether to preserve the token
		"""
		token_lower = token.lower()
		
		# Preserve emotionally significant words
		if token_lower in self.emotional_preserves:
			return True
		
		# Preserve comparative adjectives (better, worse, etc.)
		if pos_tag.startswith('JJR') or pos_tag.startswith('JJS'):
			return True
		
		# Preserve certain adverbs that indicate intensity
		if pos_tag.startswith('RB') and token_lower.endswith('ly'):
			intensity_adverbs = {'extremely', 'incredibly', 'absolutely', 'completely', 'totally'}
			if any(token_lower.startswith(adv[:4]) for adv in intensity_adverbs):
				return True
		
		return False
	
	def lemmatize(self, tokens, conservative_mode=True) -> list[str]:
		"""
		Lemmatize tokens with sentiment-aware processing.
		
		Args:
			tokens (list[str]): List of word tokens
			conservative_mode (bool): If True, preserves more emotional nuances
			
		Returns:
			list[str]: Lemmatized tokens optimized for sentiment analysis
		"""
		if not isinstance(tokens, list):
			return []
		
		# Step 1: Handle contractions
		processed_tokens = self.handle_contractions(tokens)
		
		# Step 2: Normalize expressive forms
		processed_tokens = [self.normalize_expressive_forms(token) for token in processed_tokens]
		
		# Step 3: POS tagging
		pos_tags = nltk.pos_tag(processed_tokens)
		
		# Step 4: Lemmatization with preservation logic
		lemmatized_tokens = []
		
		for token, pos in pos_tags:
			if conservative_mode and self.should_preserve_token(token, pos):
				# Preserve the token as-is
				lemmatized_tokens.append(token)
			else:
				# Apply lemmatization
				wordnet_pos = self.get_wordnet_pos(pos)
				lemmatized_token = self.lemmatizer.lemmatize(token, wordnet_pos)
				lemmatized_tokens.append(lemmatized_token)
		
		return lemmatized_tokens