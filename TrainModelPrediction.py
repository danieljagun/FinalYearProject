import requests
import pandas as pd
import os
import joblib
import praw
import time
from textblob import TextBlob

coins_list = ['Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Dogecoin']

model_folder = 'combinedModels'

# Storing loaded models
coin_models = {}

for coin in coins_list:
    coin_model_path = os.path.join(model_folder, f'{coin.lower()}_model.joblib')
    coin_models[coin] = joblib.load(coin_model_path)

def get_current_coin_data(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true' \
          f'&community_data=false&developer_data=false&sparkline=false'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching data for {coin_id}. Status code: {response.status_code}")
        return None

    data = response.json()
    return data

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
        sentiment = TextBlob(title).sentiment.polarity

        data_set = {
            "Coin": subreddit,
            "Category": category,
            "Count": count,
            "Title": title,
            "Timestamp": timestamp,
            "Link": 'https://www.reddit.com' + post.permalink,
            "Sentiment": sentiment
        }
        total_posts.append(data_set)

    return total_posts

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

        print(new_coin_data[['Prices', 'MrkCap', 'TolVol', 'Count', 'Sentiment']])

        # Making predictions using the model for the current coin
        coin_prediction = coin_models[coin].predict(new_coin_data)

        # Up or Down
        coin_prediction_label = 'Up' if coin_prediction[0] == 1 else 'Down'

        print(f"The predicted price movement for {coin} is: {coin_prediction_label}")
        print("\n" + "=" * 50 + "\n")

        time.sleep(2)

    except KeyError as e:
        print(f"KeyError in extracting data for {coin}: {e}")
        print(f"Response structure: {coin_data}")
        continue
