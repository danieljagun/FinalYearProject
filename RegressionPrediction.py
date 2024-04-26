import os

import joblib
import pandas as pd
import praw
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the RoBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")

CRYPTOCURRENCIES = ['Bitcoin', 'Ethereum', 'Solana', 'Dogecoin', 'Cardano']
MODEL_FOLDER = 'CombinedXGBoostModels'


def get_current_coin_data(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id.lower()}?localization=false&tickers=false&market_data=true'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            'CurrentPrice': data['market_data']['current_price']['usd'],
            'MarketCap': data['market_data']['market_cap']['usd'],
            'TotalVolume': data['market_data']['total_volume']['usd']
        }
    except requests.RequestException as e:
        print(f"Error fetching data for {coin_id}: {str(e)}")
        return None


def get_latest_reddit_data(subreddit, category='hot', limit=10):
    user_agent = "Scraper 1.0 by /u/danieljagun"
    try:
        reddit = praw.Reddit(client_id="K-AfR5cf3qvXEXM_MTn7IQ", client_secret="JdAlNMb6dPkWCd4bX4PdXCwIpbA5gQ",
                             user_agent=user_agent)
        subreddit_instance = reddit.subreddit(subreddit)
        posts = getattr(subreddit_instance, category)(limit=limit)
        sentiment_scores = []

        for post in posts:
            title = post.title
            inputs = tokenizer(title, return_tensors="pt", max_length=512, truncation=True)
            logits = model(**inputs).logits
            sentiment_score = logits.softmax(dim=1).detach().numpy()[0][1]
            sentiment_scores.append(sentiment_score)

        if sentiment_scores:
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            return {'RedditSentiment': average_sentiment}
        return {'RedditSentiment': 0.5}
    except Exception as e:
        print(f"Failed to fetch or process Reddit data for {subreddit}: {str(e)}")
        return {'RedditSentiment': 0.5}


def make_price_prediction(coin_name, model_folder='CombinedXGBoostModels'):
    market_data = get_current_coin_data(coin_name)
    reddit_data = get_latest_reddit_data(coin_name)
    if not market_data or not reddit_data:
        print(f"Missing required data for making predictions for {coin_name}.")
        return None

    features = {
        'MrkCap': market_data['MarketCap'],
        'TolVol': market_data['TotalVolume'],
        'Count': 10,
        'roberta_pos': reddit_data['RedditSentiment']
    }

    features_df = pd.DataFrame([features])
    model_path = os.path.join(model_folder, f'{coin_name.lower()}_combined_xgboost_model.joblib')
    model = joblib.load(model_path)
    predicted_price = model.predict(features_df)[0]

    # Convert from np.float32 to regular float for JSON serialization
    converted_price = float(predicted_price)
    print(f"Predicted future price for {coin_name}: {converted_price}")
    return converted_price  # Return the converted price


for coin in CRYPTOCURRENCIES:
    make_price_prediction(coin, MODEL_FOLDER)

