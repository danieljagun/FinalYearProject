import os
import time
from datetime import datetime, timedelta
import sys

import joblib
import pandas as pd
import praw
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

current_date = datetime.now().strftime('%Y-%m-%d')
three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')


# Function to get sentiment from news using RoBERTa
def get_news_sentiment_roberta(api_key_param, coin_name, tokenizer, model):
    news_api_endpoint = 'https://newsapi.org/v2/everything?q={}&apiKey={}&from={}&to={}&language=en'.format(
        coin_name, api_key_param, three_days_ago, current_date)
    response = requests.get(news_api_endpoint)

    if response.status_code != 200:
        print("Error fetching news data for {}. Status code: {}".format(coin_name, response.status_code))
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
coins_list = ['bitcoin', 'ethereum', 'solana', 'dogecoin', 'cardano', 'avalanche']

# Define the folder containing the models
model_folder = 'combinedModels'

# Storing loaded models
coin_models = {}

# Load models for each coin
for coin in coins_list:
    coin_model_path = os.path.join(model_folder, '{}_model.joblib'.format(coin.lower()))
    coin_models[coin] = joblib.load(coin_model_path)

# News API key
news_api_key = '2f373d0c2b6e42cbaa303f67f2f2481b'


# Function to format number as currency
def format_currency(value):
    if value < 1:
        return "${:,.4f}".format(value)
    else:
        return "${:,.2f}".format(value)


# Function to fetch current coin data
def get_current_coin_data(coin_id):
    url = 'https://api.coingecko.com/api/v3/coins/{}?localization=false&tickers=false&market_data=true' \
          '&community_data=false&developer_data=false&sparkline=false'.format(coin_id)
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching data for {}. Status code: {}".format(coin_id, response.status_code))
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


# Get the coin name from command line argument
coin_name = sys.argv[1].lower()


# Ensure the provided coin name is in the coins_list
def capitalize_name(name):
    return ' '.join(word.capitalize() for word in name.split())


if coin_name not in coins_list:
    print("Invalid coin name: {}".format(capitalize_name(coin_name)))
    exit(1)

# Getting current coin data
coin_data = get_current_coin_data(coin_name.lower())

if coin_data is None:
    exit(1)

# Getting recent posts
coin_reddit_data = fetch_reddit_data(subreddit=coin_name, category='hot', limit=10)

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
    print("Current {} price: {}".format(capitalize_name(coin_name), format_currency(new_coin_data['Prices'][0])))
    # print("MrkCap: {}".format(format_currency(new_coin_data['MrkCap'][0])))
    # print("TolVol: {}".format(format_currency(new_coin_data['TolVol'][0])))

    # Display the "Overall Feeling of" line based on the news sentiment
    overall_feeling = ''
    if new_coin_data['Sentiment'][0] > 0:
        overall_feeling = 'Happy'
    elif new_coin_data['Sentiment'][0] < 0:
        overall_feeling = 'Sad'
    else:
        overall_feeling = 'Neutral'
    print("Overall Feeling of {} from Current News Articles: {}".format(capitalize_name(coin_name), overall_feeling))

    # Making predictions using the model for the current coin
    coin_prediction = coin_models[coin_name].predict(
        new_coin_data[['Prices', 'MrkCap', 'TolVol', 'Count', 'Sentiment']])

    # Up or Down
    coin_prediction_label = 'Up' if coin_prediction[0] == 1 else 'Down'

    print("The predicted price movement for {} is: {}".format(capitalize_name(coin_name), coin_prediction_label))
    print("\n" + "=" * 50 + "\n")

    time.sleep(2)

except KeyError as e:
    print("KeyError in extracting data for {}: {}".format(coin_name, e))
    print("Response structure: {}".format(coin_data))
    exit(1)
