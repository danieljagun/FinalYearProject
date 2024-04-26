import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


def train_reddit_model(subreddit_name):
    # Load Reddit sentiment data for the cryptocurrency
    reddit_data_path = f'reddit_data/{subreddit_name.lower()}_reddit_data.csv'
    reddit_data = pd.read_csv(reddit_data_path)

    # Defining a Target Variable
    reddit_data['Sentiment_Label'] = reddit_data['Sentiment']

    # Feature Engineering
    # Vectorize the 'Title' column using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = tfidf_vectorizer.fit_transform(reddit_data['Title']).toarray()

    # Combining text features with 'Count' feature
    X = pd.concat([reddit_data[['Count']], pd.DataFrame(X_text)], axis=1)
    y = reddit_data['Sentiment_Label']

    # Convert feature names to strings
    X.columns = X.columns.astype(str)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training a Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Checking Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy on Test Set: {accuracy}')

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Saving Model
    model_folder = 'redditModels'
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'{subreddit_name.lower()}_reddit_model.joblib')
    joblib.dump(model, model_path)
    print(f'Model saved to: {model_path}')


train_reddit_model('Bitcoin')
train_reddit_model('Ethereum')
train_reddit_model('Solana')
train_reddit_model('Dogecoin')
train_reddit_model('Cardano')
train_reddit_model('Avalanche')
