import os
import time
from datetime import datetime, timedelta

import joblib
import pandas as pd
import praw
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Get the current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Calculate the date three days ago
three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')


# Function to get sentiment from news using RoBERTa
def get_news_sentiment_roberta(api_key_param, coin_name, tokenizer, model):
    news_api_endpoint = f'https://newsapi.org/v2/everything?q={coin_name}&apiKey={api_key_param}&from={three_days_ago}&to={current_date}&language=en'
    response = requests.get(news_api_endpoint)

    if response.status_code != 200:
        print(f"Error fetching news data for {coin_name}. Status code: {response.status_code}")
        return None

    data = response.json()

    overall_sentiment = 0

    if 'articles' in data:
        for article in data['articles']:
            title = article.get('title', '')
            description = article.get('description', '')

            # Tokenize and process the input for RoBERTa (Title)
            title_inputs = tokenizer(title, return_tensors="pt", max_length=512, truncation=True)
            title_logits = model(**title_inputs).logits
            title_sentiment = (title_logits.argmax().item() - 1) / 2

            # Tokenize and process the input for RoBERTa (Description)
            desc_inputs = tokenizer(description, return_tensors="pt", max_length=512, truncation=True)
            desc_logits = model(**desc_inputs).logits
            desc_sentiment = (desc_logits.argmax().item() - 1) / 2

            # Use the average sentiment from title and description
            overall_sentiment += (title_sentiment + desc_sentiment) / 2

        if data['totalResults'] > 0:
            overall_sentiment /= data['totalResults']

    return overall_sentiment


# Load the RoBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")

# Define the list of coins
coins_list = ['Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Dogecoin']

# Define the folder containing the models
model_folder = 'combinedModels'

# Storing loaded models
coin_models = {}

# Load models for each coin
for coin in coins_list:
    coin_model_path = os.path.join(model_folder, f'{coin.lower()}_model.joblib')
    coin_models[coin] = joblib.load(coin_model_path)

# News API key (not used in this version)
news_api_key = '2f373d0c2b6e42cbaa303f67f2f2481b'  # Replace with your actual News API key


# Function to fetch current coin data
def get_current_coin_data(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true' \
          f'&community_data=false&developer_data=false&sparkline=false'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching data for {coin_id}. Status code: {response.status_code}")
        return None

    data = response.json()
    return data


# Function to fetch Reddit data
def fetch_reddit_data(subreddit, category='hot', limit=10):
    user_agent = "Scraper 1.0 by /u/danieljagun"
    reddit = praw.Reddit(client_id="K-AfR5cf3qvXEXM_MTn7IQ",
                         client_secret="JdAlNMb6dPkWCd4bX4PdXCwIpbA5gQ",
                         user_agent=user_agent)

    subreddit_instance = reddit.subreddit(subreddit)
    posts = getattr(subreddit_instance, category)(limit=limit)

    total_posts = []

    for count, post in enumerate(posts, start=1):
        title = post.title
        timestamp = post.created_utc

        # Tokenize and process the input for RoBERTa
        inputs = tokenizer(title, return_tensors="pt", max_length=512, truncation=True)

        # Get the logits from the model
        logits = model(**inputs).logits

        # Use the predicted label as sentiment
        predicted_label = logits.argmax().item()

        # Scale the sentiment to be between -1 and 1
        sentiment = (predicted_label - 1) / 2

        data_set = {
            "Coin": subreddit,
            "Category": category,
            "Count": count,
            "Title": title,
            "Timestamp": timestamp,
            "Link": 'https://www.reddit.com' + post.permalink,
            "Sentiment": sentiment,
            "roberta_neg": None,
            "roberta_neu": None,
            "roberta_pos": None
        }
        total_posts.append(data_set)

    return total_posts


# Loop through each coin
for coin in coins_list:
    # Getting current coin data
    coin_data = get_current_coin_data(coin.lower())

    if coin_data is None:
        continue

    # Getting recent posts
    coin_reddit_data = fetch_reddit_data(subreddit=coin, category='hot', limit=10)

    coin_reddit_df = pd.DataFrame(coin_reddit_data)

    try:
        new_coin_data = pd.DataFrame({
            'Prices': [coin_data['market_data']['current_price']['usd']],
            'MrkCap': [coin_data['market_data'].get('market_cap', {}).get('usd', None)],
            'TolVol': [coin_data['market_data'].get('total_volume', {}).get('usd', None)],
            'Count': [coin_reddit_df['Count'].mean()],
            'Sentiment': [coin_reddit_df['Sentiment'].mean()]
        })

        # Display individual columns with additional information
        print(f"Current {coin} price: {new_coin_data['Prices'][0]}")
        print(f"MrkCap: {new_coin_data['MrkCap'][0]}")
        print(f"TolVol: {new_coin_data['TolVol'][0]}")

        # Display the "Overall Feeling of" line based on the news sentiment
        overall_feeling = ''
        if new_coin_data['Sentiment'][0] > 0:
            overall_feeling = 'Happy'
        elif new_coin_data['Sentiment'][0] < 0:
            overall_feeling = 'Sad'
        else:
            overall_feeling = 'Neutral'
        print(f"Overall Feeling of {coin}: {overall_feeling}")

        # Making predictions using the model for the current coin
        coin_prediction = coin_models[coin].predict(new_coin_data[['Prices', 'MrkCap', 'TolVol', 'Count', 'Sentiment']])

        # Up or Down
        coin_prediction_label = 'Up' if coin_prediction[0] == 1 else 'Down'

        print(f"The predicted price movement for {coin} is: {coin_prediction_label}")
        print("\n" + "=" * 50 + "\n")

        time.sleep(2)

    except KeyError as e:
        print(f"KeyError in extracting data for {coin}: {e}")
        print(f"Response structure: {coin_data}")
        continue
