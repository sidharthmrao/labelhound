"""
Evaluate sentiments for excel data
"""
from utils import read_model, load_encoder, evaluate_multiple_strings, read_pickle_model
import pandas as pd
import tensorflow_text

model_path = "models/model_3.pickle"
data_path = "examples/to_test.xlsx"
data_save = "examples/tested.xlsx"

model = read_pickle_model(model_path)
encoder = load_encoder()

data = pd.read_excel(data_path)[['tweet', 'date', 'user']]
evaluated = evaluate_multiple_strings(model, encoder, data['tweet'])

for i, val in enumerate(evaluated):
    print(val)
    if abs(val[0]-val[1]) < .3:
        evaluated[i] = "NEUTRAL"
    else:
        evaluated[i] = "POSITIVE" if val.argmax() == 1 else "NEGATIVE"

data['sentiment'] = evaluated
data.to_excel(data_save)

print("Finished evaluating data. Results saved to", data_save)
