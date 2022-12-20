from statistics import mean

import nltk
import pandas as pd
from random import shuffle
import math

data = pd.read_csv('training_data/data.csv', encoding='latin-1')

# Sentiments: 0 = Negative, 2 = Neutral, 4 = Positive
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Get rid of unnecessary columns
data = data.drop(['id', 'date', 'query', 'user'], axis=1)

# Split the data into positive and negative sets based on sentiment and isolate the text
positive_data = list(data[data['sentiment'] == 4]['text'])
negative_data = list(data[data['sentiment'] == 0]['text'])

# Save space lol
del data

# Shuffle the data
shuffle(positive_data)
shuffle(negative_data)

train_split = .7
training_positive = positive_data[:int(len(positive_data) * train_split)]
training_negative = negative_data[:int(len(negative_data) * train_split)]
testing_positive = positive_data[int(len(positive_data) * train_split):]
testing_negative = negative_data[int(len(negative_data) * train_split):]

training_data = [(tweet, 'positive') for tweet in training_positive] + \
                [(tweet, 'negative') for tweet in training_negative]

testing_data = [(tweet, 'positive') for tweet in testing_positive] + \
               [(tweet, 'negative') for tweet in testing_negative]

# Shuffle the data again
shuffle(training_data)
shuffle(testing_data)


def find_wanted_words(positive_data, negative_data):
    unwanted = nltk.corpus.stopwords.words("english")
    unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

    training_positive_words = [word.lower() for sentence in positive_data for word in nltk.word_tokenize(sentence)]
    training_negative_words = [word.lower() for sentence in negative_data for word in nltk.word_tokenize(sentence)]

    def skip_unwanted(pos_tuple):
        word, tag = pos_tuple
        if not word.isalpha() or word in unwanted:
            return False
        if tag.startswith("NN"):
            return False
        return True

    positive_words = [word for word, tag in filter(
        skip_unwanted,
        nltk.pos_tag(training_positive_words)
    )]
    negative_words = [word for word, tag in filter(
        skip_unwanted,
        nltk.pos_tag(training_negative_words)
    )]

    return positive_words, negative_words


def find_frequency_dist(positive_words, negative_words):
    positive_fd = nltk.FreqDist(positive_data)
    negative_fd = nltk.FreqDist(negative_data)

    common_set = set(positive_fd).intersection(negative_fd)

    for word in common_set:
        del positive_fd[word]
        del negative_fd[word]

    top_100_positive = {word for word, count in positive_fd.most_common(100)}
    top_100_negative = {word for word, count in negative_fd.most_common(100)}

    return top_100_positive, top_100_negative


def extract_features(sentiment_analyzer, data, top_100):
    features = dict()
    wordcount = 0
    compound_scores = list()
    positive_scores = list()

    for sentence in nltk.sent_tokenize(data):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100:
                wordcount += 1
        compound_scores.append(sentiment_analyzer.polarity_scores(sentence)["compound"])
        positive_scores.append(sentiment_analyzer.polarity_scores(sentence)["pos"])

    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later don't work with negative numbers.
    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["wordcount"] = wordcount

    return features

print("Finding wanted words...")
wanted_words = find_wanted_words(training_positive, training_negative)
print("Finding frequency distribution...")
top_100 = find_frequency_dist(*wanted_words)

print(top_100)
