import pandas as pd
from collections import defaultdict
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import nltk
nltk.data.path.append('C:/Users/janki/AppData/Roaming/nltk_data')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

df = pd.read_csv('../1-exploratory-analysis/data/PROMISE_exp.csv', sep=',', header=0, quotechar = '"', doublequote=True)

# remove the project information as the project does not ulitize it at all
del df['ProjectID']

def process_requirement_text(text):
	tokens = word_tokenize(text.lower())

	resulting_words = []
	lemmatizer = WordNetLemmatizer()

	english_stopwords = stopwords.words('english')
	tag_map = defaultdict(lambda : wn.NOUN)
	tag_map['J'] = wn.ADJ
	tag_map['V'] = wn.VERB
	tag_map['R'] = wn.ADV

	for word, tag in pos_tag(tokens):
		if word.isalpha() and word not in english_stopwords:
			resulting_words.append(
				lemmatizer.lemmatize(word, tag_map[tag[0]]))

	return ' '.join(resulting_words)

df['RequirementText'] = df['RequirementText'].apply(process_requirement_text)

df.to_csv('./output/dataset_normalized.csv', sep=',', header=True, index=False, quotechar = '"', doublequote=True)
