from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pandas as pd
import numpy as np
import json 
import os
import time

local_path = None
def load_model(input_dir):
  """
  Modify this method to deserialize you model if this environment's standard model
  loader cannot. For example, if your custom model archive contains multiple pickle
  files, you must explicitly load which ever one corresponds to your serialized model
  here.

  Returns
  -------
  object, the deserialized model
  """
  global local_path
  local_path = input_dir
  return keras.models.load_model(os.path.join(input_dir, 'imdbsentiment_model.h5'))

tokenizer = None
def load_tokenizer():
  global tokenizer
  if tokenizer is None:
    with open(os.path.join(local_path, 'tokenizer.json')) as f:
      data = json.load(f)
      tokenizer = tokenizer_from_json(data)

def score(data, model, **kwargs):
  """
  Modify this method to add pre and post processing for scoring calls. For example, this can be
  used to implement one-hot encoding for models that don't include it on their own.

  Parameters
  ----------
  data: pd.DataFrame
  model: Callable[[pd.DataFrame], pd.DataFrame]

  Returns
  -------
  pd.DataFrame
  """
  # Execute any steps you need to do before scoring
  load_tokenizer()
  if tokenizer is None:
    raise Exception("The tokenizer was not loaded.")
  all_preds = []
  for df in batches(data, 1000):
    prepped_data = []
    for _, row in df.iterrows():
      review_text = row['text']
      if review_text is None or pd.isna(review_text):
        review_text = "positive"
      sequences = tokenizer.texts_to_sequences(review_text)
      padded = pad_sequences(sequences, maxlen=1024)
      prepped_data.append(padded[0])

    # This method makes predictions against the raw, deserialized model
    if len(prepped_data) > 0:
      start_predict  = time.time_ns()
      predictions = model.predict([prepped_data], steps=1)
      print_timing("prediction", start_predict, time.time_ns())
      all_preds.append(predictions) 
    else:
      return pd.DataFrame(columns=['positive','negative'])

  # Execute any steps you need to do after scoring
  # Note: To properly send predictions back to DataRobot, the returned DataFrame should contain a
  # column for each output label for classification or a single value column for regression
  preds = [item for sublist in all_preds for item in sublist]
  results_with_prob = []
  for result in preds:
    result = result[0]
    neg_result = 1-result
    results_with_prob.append([result, neg_result])
  proba_results = np.array(results_with_prob)
  return pd.DataFrame(proba_results,columns=['positive','negative'])

def batches(seq, size):
  for pos in range(0, len(seq), size):
      yield seq.iloc[pos:pos + size]

def print_timing(tag, start, end):
  total = end - start
  seconds = total * 1e-9
  print(f"...{tag} took {seconds} seconds to execute...\n")
# def model_predict(data):
#   model = keras.models.load_model('imdbsentiment_model.h5')
#   return model.predict(data, steps=1)

# data = {'text': ["This movie was utter bullshit. Terrible", "This was the best most amazing movie ever"]}
# df = pd.DataFrame.from_dict(data)
# print(predict(df, model_predict))