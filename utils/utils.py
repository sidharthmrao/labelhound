"""
Utilities for DATAHOUND Training and Evaluation
"""
import logging
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import pickle

utils_logger = logging.getLogger('utils')


def load_encoder():
    """
    Load sentence encoder
    :return: sentence encoder
    :rtype: tf.keras.Model
    """
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


def write_model(model, path: str):
    """
    Write model to file
    :param model: model to write
    :type model: Tensorflow Model
    :param path: name of the model
    :type path: str
    """
    model.save(path)
    utils_logger.info(f'Model successfully saved to {path}')


def read_model(path: str):
    """
    Read model from file
    :param path: file path of the model
    :type path: str
    :return: model
    :rtype: Tensorflow Model
    """
    model = tf.keras.models.load_model(path)
    utils_logger.info(f'Model successfully loaded from {path}')
    return model


def read_pickle_model(path: str):
    """
    Read model from pickle file
    :param path: path to pickle file
    :type path: str
    :return: model
    :rtype: Tensorflow
    """
    file = open(path, 'rb')
    model = pickle.load(file)
    file.close()

    return model


def excel_to_dataframe(path):
    """
    Read Excel file and convert to dataframe
    :param path: path to Excel file
    :type path: str
    :return: dataframe
    :rtype: pandas.DataFrame
    """
    df = pd.read_excel(path)
    utils_logger.info(f'Excel file successfully read from {path}')
    return df


def dataframe_to_excel(df, path):
    """
    Convert dataframe to Excel file
    :param df: dataframe
    :type df: pandas.DataFrame
    :param path: path to Excel file
    :type path: str
    """
    df.to_excel(path)
    utils_logger.info(f'Excel file successfully written to {path}')


def evaluate_multiple_strings(model: tf.keras.Model, sentence_encoder: tf.keras.Model, strings: list):
    """
    Evaluate multiple strings
    :param model: model to evaluate
    :type model: tf.keras.Model
    :param sentence_encoder: sentence encoder to use
    :type sentence_encoder: tf.keras.Model
    :param strings: strings to evaluate
    :type strings: list
    :return: list of predictions
    :rtype: list
    """
    data_eval = np.array(strings)

    to_test = []
    for r in data_eval:
        emb = sentence_encoder(r)
        review_emb = tf.reshape(emb, [-1]).numpy()
        to_test.append(review_emb)
    to_test = np.array(to_test)

    return list(model.predict(to_test))


def evaluate_string(model: tf.keras.Model, sentence_encoder: tf.keras.Model, string: str):
    """
    Evaluate a single string
    :param model: model to evaluate
    :type model: tf.keras.Model
    :param sentence_encoder: sentence encoder to use
    :type sentence_encoder: tf.keras.Model
    :param string: string to evaluate
    :type string: str
    :return: prediction
    :rtype: list
    """
    return evaluate_multiple_strings(model, sentence_encoder, [string])
