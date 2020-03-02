import setuptools

setuptools.setup(
  name="Profanity",
  packages=setuptools.find_packages(),
  package_data={ 'Profanity': ['data/model.joblib', 'data/vectorizer.joblib'] },
)
