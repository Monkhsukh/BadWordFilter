
# Profanity

## Data

A combined dataset from two publicly-available sources:

-   the “Twitter” dataset from  [t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data), which contains tweets scraped from Twitter.
-   the “Wikipedia” dataset from  [this Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  published by Alphabet’s  [Conversation AI](https://conversationai.github.io/)  team, which contains comments from Wikipedia’s talk page edits.

Here’s what my dataset ended up looking like:

![](https://cdn-images-1.medium.com/max/1600/1*Bw_we8cbs-WOpWXOCxzSTg.png)

Combined = Tweets + Wikipedia

## Text clean
Text normalization includes:
-   converting all letters to lower or upper case
-   converting numbers into words or removing numbers
-   removing punctuations, accent marks and other diacritics
-   removing white spaces
-   expanding abbreviations
-   removing stop words, sparse terms, and particular words
-   text  canonicalization

## Convert text to lowercase

**Example 1. Convert text to lowercase**

**Python code:**
```python
input_str = ”The 5 biggest countries by population in 2017 are China, India, United States, Indonesia, and Brazil.”  
input_str = input_str.lower()  
print(input_str)
```
**Output:**

    the 5 biggest countries by population in 2017 are china, india, united states, indonesia, and brazil.

## Remove numbers

Remove numbers if they are not relevant to your analyses. Usually, regular expressions are used to remove numbers.

**Example 2. Numbers removing**

**Python code:**

```python
import re  
input_str = ’Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls.’  
result = re.sub(r’\d+’, ‘’, input_str)  
print(result)
```
**Output:**

	Box A contains red and white balls, while Box B contains red and blue balls.

## Remove punctuation

The following code removes this set of symbols [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:

**Example 3. Punctuation removal**

**Python code:**

```python
import string  
input_str = “This &is [an] example? {of} string. with.? punctuation!!!!” # Sample string  
result = input_str.translate(string.maketrans(“”,””), string.punctuation)  
print(result)
```
**Output:**

This is an example of string with punctuation

## Remove whitespaces

To remove leading and ending spaces, you can use the  _strip()_  function:

**Example 4. White spaces removal**

**Python code:**
```python
input_str = “ \t a string example\t “  
input_str = input_str.strip()  
input_str
```
**Output:**

	‘a string example’
## Tokenization

Tokenization is the process of splitting the given text into smaller pieces called tokens. Words, numbers, punctuation marks, and others can be considered as tokens. In  [this table](https://docs.google.com/spreadsheets/d/1-9rMhfcmxFv2V2Q5ZWn1FfLDZZYsuwb1eoSp9CiEEOg/edit?usp=sharing)  (“Tokenization” sheet) several tools for implementing tokenization are described.

## Remove stop words

“Stop words” are the most common words in a language like “the”, “a”, “on”, “is”, “all”. These words do not carry important meaning and are usually removed from texts. It is possible to remove stop words using  [Natural Language Toolkit (NLTK)](http://www.nltk.org/), a suite of libraries and programs for symbolic and statistical natural language processing.

**Example 7. Stop words removal**

**Code:**
```python
input_str = “NLTK is a leading platform for building Python programs to work with human language data.”  
stop_words = set(stopwords.words(‘english’))  
from nltk.tokenize import word_tokenize  
tokens = word_tokenize(input_str)  
result = [i for i in tokens if not i in stop_words]  
print (result)
```
**Output:**

[‘NLTK’, ‘leading’, ‘platform’, ‘building’, ‘Python’, ‘programs’, ‘work’, ‘human’, ‘language’, ‘data’, ‘.’]

A  [scikit-learn](http://scikit-learn.org/stable/)  tool also provides a stop words list:

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

It’s also possible to use  [spaCy](https://spacy.io/), a free open-source library:

from spacy.lang.en.stop_words import STOP_WORDS

## Remove sparse terms and particular words

In some cases, it’s necessary to remove sparse terms or particular words from texts. This task can be done using stop words removal techniques considering that any group of words can be chosen as the stop words.

## Stemming

Stemming is a process of reducing words to their word stem, base or root form (for example, books — book, looked — look). The main two algorithms are  [Porter stemming algorithm](https://tartarus.org/martin/PorterStemmer/)  (removes common morphological and inflexional endings from words  [[14])](https://tartarus.org/martin/PorterStemmer/)  and  [Lancaster stemming algorithm](http://web.archive.org/web/20140827005744/http:/www.comp.lancs.ac.uk/computing/research/stemming/index.htm)  (a more aggressive stemming algorithm). In the  [“Stemming” sheet of the table](https://docs.google.com/spreadsheets/d/1-9rMhfcmxFv2V2Q5ZWn1FfLDZZYsuwb1eoSp9CiEEOg/edit?usp=sharing)  some stemmers are described.

**Example 8. Stemming using NLTK:**

**Code:**
```python
from nltk.stem import PorterStemmer  
from nltk.tokenize import word_tokenize  
stemmer= PorterStemmer()  
input_str=”There are several types of stemming algorithms.”  
input_str=word_tokenize(input_str)  
for word in input_str:  
    print(stemmer.stem(word))
```
**Output:**

	There are sever type of stem algorithm.
## Lemmatization

The aim of lemmatization, like stemming, is to reduce inflectional forms to a common base form. As opposed to stemming, lemmatization does not simply chop off inflections. Instead it uses lexical knowledge bases to get the correct base forms of words.

Lemmatization tools are presented libraries described above:  [NLTK (WordNet Lemmatizer)](http://www.nltk.org/_modules/nltk/stem/wordnet.html),  [spaCy](https://spacy.io/api/lemmatizer),  [TextBlob](http://textblob.readthedocs.io/en/dev/quickstart.html#words-inflection-and-lemmatization),  [Pattern](https://www.clips.uantwerpen.be/pages/pattern-en#conjugation),  [gensim](https://radimrehurek.com/gensim/utils.html),  [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/simple.html),  [Memory-Based Shallow Parser (MBSP)](https://www.clips.uantwerpen.be/pages/MBSP#lemmatizer),  [Apache OpenNLP](https://opennlp.apache.org/docs/1.8.4/manual/opennlp.html#tools.lemmatizer.tagging.cmdline),  [Apache Lucene](http://lucene.apache.org/core/),  [General Architecture for Text Engineering (GATE)](https://gate.ac.uk/),  [Illinois Lemmatizer](https://cogcomp.org/page/software_view/illinois-lemmatizer), and  [DKPro Core](https://dkpro.github.io/dkpro-core/releases/1.8.0/docs/component-reference.html#_lemmatizer).

**Example 9. Lemmatization using NLTK:**

**Code:**
```python
from nltk.stem import WordNetLemmatizer  
from nltk.tokenize import word_tokenize  
lemmatizer=WordNetLemmatizer()  
input_str=”been had done languages cities mice”  
input_str=word_tokenize(input_str)  
for word in input_str:  
    print(lemmatizer.lemmatize(word))
```
**Output:**

`be have do language city mouse`

## Training

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Read in data
data = pd.read_csv('clean_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(texts)

# Train the model
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)

# Save the model
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(cclf, 'model.joblib')
```

Two major steps are happening here: (1) vectorization and (2) training.

### Vectorization: Bag of Words

I used  `scikit-learn`’s  [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)  class, which basically turns any text string into a vector by counting how many times each given word appears. This is known as a  [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model)  (BOW) representation. For example, if the only words in the English language were  `the`,  `cat`,  `sat`, and  `hat`, a possible vectorization of the sentence  `the cat sat in the hat`  might be:

![](https://cdn-images-1.medium.com/max/1600/1*sbnts1u_QFB_V-X5DSC3pg.png)

“the cat sat in the hat” -> [2, 1, 1, 1, 1]

The `???`  represents any unknown word, which for this sentence is  `in`. Any sentence can be represented in this way as counts of  `the`,  `cat`,  `sat`,  `hat`, and `???`!

![](https://cdn-images-1.medium.com/max/1600/1*-wONWZDab2gNQP3Rfdpt_A.png)

A handy reference table for the next time you need to vectorize “cat cat cat cat cat”

Of course, there are far more words in the English language, so in the code above I use the  `fit_transform()`  method, which does 2 things:

-   **Fit:**  learns a vocabulary by looking at all words that appear in the dataset.
-   **Transform**: turns each text string in the dataset into its vector form.

### [](https://victorzhou.com/blog/better-profanity-detection-with-scikit-learn/#training-linear-svm)Training: Linear SVM

The model I decided to use was a Linear Support Vector Machine (SVM), which is implemented by  `scikit-learn`’s  [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)  class.  [This post](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)  and  [this tutorial](https://www.svm-tutorial.com/2014/11/svm-understanding-math-part-1/)  are good introductions if you don’t know what SVMs are.



Here’s one (simplified) way you could think about why the Linear SVM works: during the training process, the model learns which words are “bad” and how “bad” they are because those words appear more often in offensive texts.  **It’s as if the training process is picking out the “bad” words for me**, which is much better than using a wordlist I write myself!

A Linear SVM combines the best aspects of the other profanity detection libraries I found: it’s fast enough to run in real-time yet robust enough to handle many different kinds of profanity.

## Caveats

That being said,  `profanity-check`  is far from perfect. Let me be clear: take predictions from  `profanity-check`  with a grain of salt because  **it makes mistakes.**  For example, its not good at picking up less common variants of profanities like “f4ck you” or “you b1tch” because they don’t appear often enough in the training data. You’ll never be able to detect  _all_  profanity (people will come up with new ways to evade filters), but  `profanity-check`  does a good job at finding most.

