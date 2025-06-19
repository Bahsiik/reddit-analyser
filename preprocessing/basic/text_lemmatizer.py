import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Comment once done for the first time
# nltk.download('wordnet', download_dir=ntlk_dir)
# nltk.download('omw-1.4', download_dir=ntlk_dir)
# nltk.download('averaged_perceptron_tagger_eng', download_dir=ntlk_dir)

class TextLemmatizer:
	"""
	Class for lemmatizing word tokens using NLTK's WordNetLemmatizer.
	"""
	
	def __init__(self) -> None:
		self.lemmatizer = WordNetLemmatizer()
	
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
	
	def lemmatize(self, tokens) -> list[str]:
		"""
		Lemmatize a list of word tokens.
		
		Args:
			tokens (List[str]): List of word tokens
		
		Returns:
			List[str]: Lemmatized tokens
		"""
		if not isinstance(tokens, list):
			return []

		pos_tags = nltk.pos_tag(tokens)  # POS tagging
		return [
			self.lemmatizer.lemmatize(token, self.get_wordnet_pos(pos))
			for token, pos in pos_tags
		]
