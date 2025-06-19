from nltk.corpus import stopwords

class StopwordRemover:
	"""
	Lightweight stopword remover optimized for sentiment analysis.
	Preserves emotionally significant words while removing neutral noise.
	"""
	
	def __init__(self, language="english") -> None:
		"""
		Initialize the stopword remover with a given language.
		
		Args:
			language (str): Language of the stopwords (default is 'english')
		"""
		self.base_stop_words = set(stopwords.words(language))
		
		# Define words to preserve even if they're in standard stopwords
		self.emotional_preserves = {
			# Intensity markers
			"very", "really", "quite", "extremely", "absolutely", "totally",
			"so", "too", "such", "much", "more", "most", "less", "least",
			
			# Personal expression (important for sentiment ownership)
			"i", "me", "my", "myself", "you", "your", "yourself",
			
			# Sentiment connectors (change meaning/context)
			"but", "however", "although", "though", "because", "since", "as",
			
			# Temporal context (important for sentiment timing)
			"now", "still", "yet", "already", "just"
		}
		
		# Extended negations (crucial for sentiment)
		self.negations = {
			# Standard negations
			"not", "no", "never", "n't",
			
			# Additional negative forms
			"neither", "nor", "none", "nothing", "nobody", "nowhere",
			
			# Partial negations
			"hardly", "barely", "scarcely",
			
			# Implicit negations
			"without", "lack", "fail"
		}
		
		# Create the final stopwords set
		self.stop_words = self.base_stop_words - self.emotional_preserves - self.negations
	
	def remove_stopwords(self, tokens, keep_negation=True) -> list[str]:
		"""
		Remove stopwords while preserving emotionally significant words.
		
		Args:
			tokens (List[str]): List of word tokens
			keep_negation (bool): Whether to preserve negation words (default: True)
		
		Returns:
			List[str]: Tokens without neutral stopwords but keeping emotional indicators
		"""
		if not isinstance(tokens, list):
			return []
		
		if keep_negation:
			# Use our optimized stopword list that already preserves negations and emotional words
			return [word for word in tokens if word.lower() not in self.stop_words]
		else:
			# If negation preservation is disabled, only preserve emotional words (not negations)
			preserve_set = self.emotional_preserves
			return [word for word in tokens if word.lower() not in self.base_stop_words or word.lower() in preserve_set]
	
	def get_preserved_words(self) -> dict:
		"""
		Get the sets of words that are preserved for transparency.
		
		Returns:
			dict: Dictionary containing the different categories of preserved words
		"""
		return {
			"emotional_preserves": self.emotional_preserves,
			"negations": self.negations,
			"total_preserved": self.emotional_preserves | self.negations
		}
	
	def get_stopwords_count(self) -> dict:
		"""
		Get statistics about stopwords filtering.
		
		Returns:
			dict: Statistics about original vs filtered stopwords
		"""
		return {
			"original_stopwords": len(self.base_stop_words),
			"preserved_words": len(self.emotional_preserves | self.negations),
			"final_stopwords": len(self.stop_words),
			"reduction_percentage": round((len(self.emotional_preserves | self.negations) / len(self.base_stop_words)) * 100, 2)
		} 