import pkg_resources
import numpy as np
from sklearn.externals import joblib

vectorizer = joblib.load(pkg_resources.resource_filename('Profanity', 'data/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('Profanity', 'data/model.joblib'))

def _get_profane_prob(prob):
  return prob[1]

#TODO: Clean texts
def predict(texts):
  return model.predict(vectorizer.transform(texts))

def predict_prob(texts):
  return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))
