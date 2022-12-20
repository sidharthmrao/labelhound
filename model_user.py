import pickle
from new_sentiment_stuff_testing_digital_ocean import remove_noise
from nltk.tokenize import word_tokenize

print("Running the model user.")

file = open("models/model_0.pickle", "rb")
classifier = pickle.load(file)
file.close()

tokens = remove_noise(word_tokenize("I hate you!"))

print(classifier.classify(dict([token, True] for token in tokens)))
