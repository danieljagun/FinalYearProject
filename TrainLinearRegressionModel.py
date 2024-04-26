import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def prepare_features(data, vectorizer):
    X_text = vectorizer.fit_transform(data['Title']).toarray()
    if 'Count' in data:
        count_feature = StandardScaler().fit_transform(data[['Count']])
        return np.hstack((X_text, count_feature))
    return X_text


def save_model(model, vectorizer, subreddit_name):
    model_folder = ' RegressionRedditModels'
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'{subreddit_name.lower()}_reddit_xgb_model.joblib')
    joblib.dump({'model': model, 'vectorizer': vectorizer}, model_path)
    print(f'Model saved to: {model_path}')


def train_reddit_model(subreddit_name):
    reddit_data_path = f'reddit_data/{subreddit_name.lower()}_reddit_data.csv'
    reddit_data = load_data(reddit_data_path)
    if reddit_data is None:
        return

    y = reddit_data['roberta_pos'].values
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = prepare_features(reddit_data, tfidf_vectorizer)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{subreddit_name} Model - MSE: {mse}, R-squared: {r2}')
    save_model(model, tfidf_vectorizer, subreddit_name)


subreddit_names = ['Bitcoin', 'Ethereum', 'Solana', 'Dogecoin', 'Cardano', 'Avalanche']
for subreddit_name in subreddit_names:
    train_reddit_model(subreddit_name)
