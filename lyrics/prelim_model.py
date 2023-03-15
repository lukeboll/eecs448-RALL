# Common libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, train_test_split 
import matplotlib.pyplot as plt

# Linear regression Question libraries
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler



# Logistic Regression Question Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')

# Reading in the Spotify Data and conducting pre-processing
df = pd.read_csv('spotify_songs.csv')

new_df = pd.DataFrame({
    'text': df.loc[:, 'lyrics'],
    'label': df.loc[:, 'track_popularity']
}).head(1000)

nltk_stopwords = set(stopwords.words('english'))

def PreprocessSentence(sentence, nltk_stopwords):
    s = sentence.lower()
    s = word_tokenize(s)
    words = []
    for w in s:
        if w not in nltk_stopwords and len(w) > 1:
            words.append(w)
    return ' '.join(words)

def PreprocessData(new_df, nltk_stopwords):
    # Input - the dataframe, with columns label and text
    # output - the dataframe with the text processed as described earlier
    new_df.dropna(subset=['text'], inplace=True)  # drop rows with missing values in the 'text' column
    new_df.text = new_df.text.map(lambda x: PreprocessSentence(x, nltk_stopwords))
    return new_df

new_df = PreprocessData(new_df, nltk_stopwords)

""" 
Linear Model with Bag of Words 
"""

# Step 1: Prepare the data

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(new_df.text)
y = new_df['label'].to_numpy()

# Step 3: Data Transformation
X = X.toarray()
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=448)

X_train = sp.csc_matrix(X_train)
X_test = sp.csc_matrix(X_test)
model = Ridge(alpha=1000000)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mse = r2_score(y_test, y_pred)
y_pred_train = model.predict(X_train)
train_mse = r2_score(y_train, y_pred_train)
print("BoW R^2:", mse)
print("BoW Train R^2:", train_mse)

import pickle

# save
with open('model_bag_of_words.pkl','wb') as f:
    pickle.dump(model,f)


""" 
Linear Model with tf-idf
"""

# Step 1: Prepare the data

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(new_df.text)
y = new_df['label'].to_numpy()

# Step 3: Data Transformation
X = X.toarray()
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=448)
X_train = sp.csc_matrix(X_train)
X_test = sp.csc_matrix(X_test)

model = Ridge(alpha=1000000)

model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mse = r2_score(y_test, y_pred)
y_pred_train = model.predict(X_train)
train_mse = r2_score(y_train, y_pred_train)
print("Tf-idf R^2:", mse)
print("Tf-idf Training R^2:", train_mse)

# save
with open('model_tf_idf.pkl','wb') as f:
    pickle.dump(model,f)