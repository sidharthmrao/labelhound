"""
Train a new sentiment classifier using a set of training data.
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow_text
from tensorflow import keras
from utils import dataframe_to_excel


use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


ratio_to_train = .3
data_path = 'training_data/data.csv'
test_set_size = .25
model_path = "models/sentiment_classifier"

data = pd.read_csv(data_path, encoding='latin-1')

# Sentiments: 0 = Negative, 2 = Neutral, 4 = Positive
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Get rid of unnecessary columns
data = data.drop(['id', 'date', 'query', 'user'], axis=1)
data = data.sample(frac=ratio_to_train)

# One hot encode sentiment data
type_one_hot = OneHotEncoder(sparse=False).fit_transform(
    data['sentiment'].to_numpy().reshape(-1, 1)
)

# Split data into training and testing sets
train_reviews, test_reviews, y_train, y_test = train_test_split(
    data.text,
    type_one_hot,
    test_size=test_set_size
)


# Set up embeddings
X_train = []
for r in tqdm(train_reviews):
    emb = use(r)
    review_emb = tf.reshape(emb, [-1]).numpy()
    X_train.append(review_emb)
X_train = np.array(X_train)

X_test = []
for r in tqdm(test_reviews):
    emb = use(r)
    review_emb = tf.reshape(emb, [-1]).numpy()
    X_test.append(review_emb)
X_test = np.array(X_test)

# Set up model
model = keras.Sequential()
model.add(
    keras.layers.Dense(
        units=256,
        input_shape=(X_train.shape[1],),
        activation='relu'
    )
)
model.add(
    keras.layers.Dropout(rate=0.5)
)
model.add(
    keras.layers.Dense(
        units=128,
        activation='relu'
    )
)
model.add(
    keras.layers.Dropout(rate=0.5)
)
model.add(keras.layers.Dense(2, activation='softmax'))  #2
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    shuffle=True
)

# Evaluate model
model.evaluate(X_test, y_test)

dataframe_to_excel(model, model_path)
