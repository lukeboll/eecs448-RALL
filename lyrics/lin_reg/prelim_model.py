# Common libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, train_test_split 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# Linear regression Question libraries
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

# Logistic Regression Question Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

# Step 4: Model Training and Evaluation with 5-fold cross-validation
alphas = np.logspace(-1, 7, num=9) # set alpha values to iterate over
train_mses = [] # to store the training MSEs for each alpha
test_mses = [] # to store the testing MSEs for each alpha
train_mses_std = [] # to store the training MSEs for each alpha
test_mses_std = [] # to store the testing MSEs for each alpha
kf = KFold(n_splits=5, shuffle=True, random_state=448)

for alpha in tqdm(alphas, desc='BoW model'):
    train_mse_alpha = []
    test_mse_alpha = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train = sp.csc_matrix(X_train)
        X_test = sp.csc_matrix(X_test)
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        y_pred_train = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_mse_alpha.append(train_mse)
        test_mse_alpha.append(test_mse)

    train_mses.append(np.mean(train_mse_alpha))
    train_mses_std.append(np.std(train_mse_alpha))

    test_mses.append(np.mean(test_mse_alpha))
    test_mses_std.append(np.std(test_mse_alpha))

plt.figure()
plt.errorbar(alphas, train_mses, yerr=train_mses_std, label='Train', alpha=0.7)
plt.errorbar(alphas, test_mses, yerr=test_mses_std, label='Test', alpha=0.7)
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.xscale('log')
plt.title('MSE vs alpha for BoW model')
plt.legend()
plt.savefig('bag_of_words.png')

""" 
Linear Model with TF-IDF
"""

# Step 1: Prepare the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(new_df.text)
y = new_df['label'].to_numpy()

# Step 3: Data Transformation
X = X.toarray()
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)

# Step 4: Model Training and Evaluation with 5-fold cross-validation
alphas = np.logspace(-1, 7, num=9) # set alpha values to iterate over
train_mses = [] # to store the training MSEs for each alpha
test_mses = [] # to store the testing MSEs for each alpha
train_mses_std = [] # to store the training MSEs for each alpha
test_mses_std = [] # to store the testing MSEs for each alpha
kf = KFold(n_splits=5, shuffle=True, random_state=448)

for alpha in tqdm(alphas, desc='Tf-idf model'):
    train_mse_alpha = []
    test_mse_alpha = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train = sp.csc_matrix(X_train)
        X_test = sp.csc_matrix(X_test)
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        y_pred_train = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_mse_alpha.append(train_mse)
        test_mse_alpha.append(test_mse)

    train_mses.append(np.mean(train_mse_alpha))
    train_mses_std.append(np.std(train_mse_alpha))

    test_mses.append(np.mean(test_mse_alpha))
    test_mses_std.append(np.std(test_mse_alpha))

plt.figure()
plt.errorbar(alphas, train_mses, yerr=train_mses_std, label='Train', alpha=0.7)
plt.errorbar(alphas, test_mses, yerr=test_mses_std, label='Test', alpha=0.7)
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.xscale('log')
plt.title('MSE vs alpha for Tf-idf model')
plt.legend()
plt.savefig('tf-idf.png')