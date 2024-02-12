import os
import pandas as pd
import praw

from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

user_agent = "Scraper 1.0 by /u/danieljagun"
reddit = praw.Reddit(client_id="K-AfR5cf3qvXEXM_MTn7IQ",
                     client_secret="JdAlNMb6dPkWCd4bX4PdXCwIpbA5gQ",
                     user_agent=user_agent)

subreddits = ['Bitcoin', 'Ethereum', 'XRP', 'Solana', 'Cardano', 'Dogecoin']
categories = ['hot', 'new', 'rising', 'top']

for subreddit in subreddits:
    total_posts = []

    for category in categories:
        subreddit_instance = reddit.subreddit(subreddit)
        posts = getattr(subreddit_instance, category)(limit=None)
        count = 0

        for post in posts:
            count += 1
            title = post.title
            timestamp = post.created_utc

            # Sentiment Analysis (RoBERTa)
            inputs = tokenizer.encode(title, return_tensors="pt")
            outputs = model(inputs)[0]
            scores = outputs.detach().numpy()
            scores = softmax(scores, axis=1).tolist()[0]

            # Extracting sentiment probabilities
            scores_dict = {
                'roberta_neg': scores[0],
                'roberta_neu': scores[1],
                'roberta_pos': scores[2]
            }

            # Define sentiment label based on the highest probability
            sentiment_label = scores.index(max(scores))

            data_set = {
                "Subreddit": subreddit,
                "Category": category,
                "Count": count,
                "Title": title,
                "Timestamp": timestamp,
                "Link": 'https://www.reddit.com' + post.permalink,
                "Sentiment": sentiment_label
            }
            total_posts.append(data_set)

        print(f"Number of {category} headlines for {subreddit}: {count}")

    df = pd.DataFrame(total_posts)

    # Formatting the time column
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')

    folder_name = 'reddit_data'
    os.makedirs(folder_name, exist_ok=True)

    csv_filename = os.path.join(folder_name, f"{subreddit.lower()}_reddit_data.csv")
    df.to_csv(csv_filename, sep=',', index=False)

    print(f"Reddit data for {subreddit} saved to {csv_filename}")
