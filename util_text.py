# written by kylesim

import re
import pprint
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RM_SYMBOLS = r'[~!@#$%^&*()\-_=+\[{\]}\\|;:\'\",<.>/?`]'

def clean_string(string, mask_numbers=False):
	# remove all the special characters
	# SYMBOLS_32 => ~!@#$%^&*()-_=+[{]}\|;:'",<.>/?`
	#string = re.sub(r'\W', ' ', string) # _ not removed
	string = re.sub(RM_SYMBOLS, ' ', string)
	
	# substitute multiple spaces with a single space
	string = re.sub(r'\s+', ' ', string)

	if mask_numbers:
		string = re.sub(r'\d+', '00', string)

	# convert to lowercase letters
	string = string.lower()

	# remove all leading and trailing whitespaces
	return string.strip()


def count_vectorize(docs, max_features = None, min_df = 1, max_df = 1.0):
	# bag of words model (word -> number)
	vectorizer = CountVectorizer(max_features = max_features, min_df = min_df, max_df = max_df)
	numeric_docs = vectorizer.fit_transform(docs).toarray()
	feature_names = vectorizer.get_feature_names()
	stop_words = vectorizer.get_stop_words()
	print(vectorizer.vocabulary_)
	print(stop_words)
	print(feature_names)
	print(numeric_docs)


def tfidf_vectorize(docs, max_features = None, min_df = 1, max_df = 1.0):
	# bag of words model (word -> number)
	vectorizer = TfidfVectorizer(max_features = max_features, min_df = min_df, max_df = max_df)
	numeric_docs = vectorizer.fit_transform(docs).toarray()
	feature_names = vectorizer.get_feature_names()
	stop_words = vectorizer.get_stop_words()
	print(vectorizer.vocabulary_)
	print(stop_words)
	print(feature_names)
	pprint.pprint(numeric_docs)


if __name__ == "__main__":
	string = "1. this is a test string ~!@#$%^&*()-_=+[{]}\|;:'\",<.>/?`"
	docs = [
	"This is the first document.",
	"This is the second document.",
	"This is the 2nd document?",
	"This is the third document.",
	"This is the 3rd document?",
	"What is the 3rd document?",
	]
	count_vectorize(docs)
	tfidf_vectorize(docs)
