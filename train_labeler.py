"""
Train a new labeler using a set of training data.
"""
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

data = pd.read_csv('training_data/data.csv', encoding='latin-1')

vect = CountVectorizer(analyzer="word", ngram_range=(2, 2), stop_words='english')
data = vect.fit_transform(data['text'])