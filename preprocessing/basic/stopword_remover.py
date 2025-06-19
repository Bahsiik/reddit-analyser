import nltk
from nltk.corpus import stopwords

# Comment once done for the first time
# nltk.download('stopwords', download_dir=ntlk_dir)

class StopwordRemover:
	"""
	Class for removing stopwords from tokenized text.
	"""
	
	def __init__(self, language="english") -> None:
		"""
		Initialize the stopword remover with a given language.
		
		Args:
			language (str): Language of the stopwords (default is 'english')
		"""
		self.stop_words = set(stopwords.words(language))
	
	def remove_stopwords(self, tokens, keep_negation=True) -> list[str]:
		"""
		Remove stopwords from a list of tokens.
		
		Args:
			tokens (List[str]): List of word tokens
		
		Returns:
			List[str]: Tokens without stopwords
		"""
		if not isinstance(tokens, list):
			return []
		if keep_negation:
			negations = {"not", "no", "never", "n't"}
			return [word for word in tokens if word.lower() not in self.stop_words or word.lower() in negations]
		else:
			return [word for word in tokens if word.lower() not in self.stop_words]