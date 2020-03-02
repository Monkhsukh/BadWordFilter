import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

# Read in data
data = pd.read_csv('clean_data.csv')
train, test = train_test_split(data, test_size=0.2)
texts_test = test['text'].astype(str)
texts_train = train['text'].astype(str)
#print('Total rows in test is {}'.format(len(data)))
"""
# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)

#train

x_train = vectorizer.fit_transform(texts_train)
y_train = train['is_offensive']
#print(vectorizer.get_feature_names())
#print(X.toarray())

# Train the model
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(x_train, y_train)

joblib.dump(vectorizer, 'vectorizer_test.joblib')
joblib.dump(cclf, 'model_test.joblib') 
"""
vectorizer = joblib.load('vectorizer_test.joblib')
model = joblib.load('model_test.joblib')

# Predict
predict_test = model.predict(vectorizer.transform(texts_test.to_list()))

print (accuracy_score(predict_test, test['is_offensive']))

#np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))
