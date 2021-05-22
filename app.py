from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import string
def clean(text):
 text = re.sub('https?://\S+|www\.\S+', '', text)
 text = re.sub(r'\s+', ' ', text, flags=re.I)
 text = re.sub('\[.*?\]', '', text)
 text = re.sub('\n', '', text)
 text = re.sub('\w*\d\w*', '', text)
 text = re.sub('<.*?>+', '', text)
 text = str(text).lower()
 text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
 return text


def remove_stopwords(text):
 text = [word for word in text if word not in stopwords]
 return text

app = Flask(__name__)
@app.route('/')
def home():
 return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
 data = pd.read_csv("data/vaccination_tweets.csv")
 data=data[:1000]

 data['text'] = data['text'].apply(lambda x: clean(x))
 from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 analyser = SentimentIntensityAnalyzer()
 scores = []
 for i in range(len(data['text'])):
  score = analyser.polarity_scores(data['text'][i])
  score = score['compound']
  scores.append(score)
 sentiment = []
 for i in scores:
  if i >= 0:
   sentiment.append(1)
  elif i <0:
   sentiment.append(0)
 data['sentiment'] = pd.Series(np.array(sentiment))

 # Features and Labels
 df_x = data['text']
 df_y = data['sentiment']
 # Extract the features with countVectorizer
 corpus = df_x

 from nltk.stem import WordNetLemmatizer
 from nltk.tokenize import sent_tokenize, word_tokenize
 import re
 from nltk.corpus import stopwords
 # Preprocessing
 processed_texts_list = []
 for text in corpus:
  processed_text = text.lower()
  processed_text = re.sub('[^a-zA-Z]', ' ', processed_text)
  processed_text = re.sub(r'\s+', ' ', processed_text)
  processed_texts_list.append(processed_text)
 all_texts_words = []
 lemmmatizer = WordNetLemmatizer()
 for text in processed_texts_list:
  data = text
  words = word_tokenize(data)
  words = [lemmmatizer.lemmatize(word.lower()) for word in words if
           (not word in set(stopwords.words('english')) and word.isalpha())]
  all_texts_words.append(words)
 all_texts_words
 series = pd.Series(all_texts_words).astype(str)

 """
 from gensim.models import Word2Vec
 word2vec_tweet = Word2Vec(all_texts_words, vector_size=10, window=3, min_count=1, workers=4,sg=1)

 # Forming x matrix
 X = []
 for i in range(len(all_texts_words)):
  vect_doc = np.zeros(10)
  for j in range(len(all_texts_words[i])):
   vect_doc += word2vec_tweet.wv[all_texts_words[i][j]]
  vect_doc = vect_doc / (len(all_texts_words[i]))
  X.append(vect_doc)
 """
 cv = TfidfVectorizer()
 X = cv.fit_transform(series)
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33,
                                                     random_state=42)


 #Regression Logistic
 from sklearn.linear_model import LogisticRegression
 clf = LogisticRegression(penalty='none',solver='newton-cg')
 clf.fit(X_train, y_train)
 score=clf.score(X_test, y_test)
 print("******le scocre avec Logistic Regression*******",score)
 if request.method == 'POST':
  comment = request.form['comment']
 data = [comment]
 """
 comment=clean(comment)
 words = word_tokenize(comment)
 words = [lemmmatizer.lemmatize(word.lower()) for word in words if
          (not word in set(stopwords.words('english')) and word.isalpha())]
 vect_data=np.zeros(10)
 for word in words:
  vect_data += word2vec_tweet.wv[word]
 vect_data=vect_data/(len(words))
 print(vect_data)
 my_prediction = clf.predict(vect_data.reshape(1,-1))
 """
 vect = cv.transform(data).toarray()
 #print(vect)
 my_prediction = clf.predict(vect)
 return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
 app.run(host='127.0.0.1', port=5555, debug=True)
