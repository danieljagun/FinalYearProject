import os
import pandas as pd
import praw
from pymongo import MongoClient

from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

mongo_uri = "mongodb+srv://danieljagun:Daniel202@cluster.tokbwgs.mongodb.net/?retryWrites=true&w=majority"


MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

user_agent = "Scraper 1.0 by /u/danieljagun"
reddit = praw.Reddit(client_id="K-AfR5cf3qvXEXM_MTn7IQ",
                     client_secret="JdAlNMb6dPkWCd4bX4PdXCwIpbA5gQ",
                     user_agent=user_agent)

client = MongoClient(mongo_uri)
db = client['reddit_data']

subreddits = ['Bitcoin', 'Ethereum', 'Solana', 'Dogecoin', 'Cardano']

categories = ['hot', 'new', 'rising', 'top']


def normalize_subreddit_name(name):
    return "avalanche" if name.lower() == "avax" else name.lower()


for subreddit in subreddits:
    total_posts = []

    try:
        subreddit_instance = reddit.subreddit(subreddit)
    except Exception as e:
        print(f"Failed to fetch subreddit '{subreddit}': {e}")
        continue  # Skip to the next subreddit if the current one fails

    for category in categories:
        try:
            posts = getattr(subreddit_instance, category)(limit=None)
        except AttributeError:
            print(f"Category '{category}' does not exist in subreddit '{subreddit}'. Skipping.")
            continue
        except Exception as e:
            print(f"Error retrieving posts from subreddit '{subreddit}', category '{category}': {e}")
            continue

        count = 0
        for post in posts:
            count += 1
            title = post.title
            timestamp = post.created_utc

            # Sentiment Analysis (RoBERTa)
            inputs = tokenizer.encode(title, return_tensors="pt")
            outputs = model(inputs)[0]
            scores = softmax(outputs.detach().numpy(), axis=1).tolist()[0]

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
                "Sentiment": sentiment_label,
                "roberta_neg": scores_dict['roberta_neg'],
                "roberta_neu": scores_dict['roberta_neu'],
                "roberta_pos": scores_dict['roberta_pos']
            }
            total_posts.append(data_set)

        print(f"Number of {category} headlines for {subreddit}: {count}")

    df = pd.DataFrame(total_posts)

    # Formatting the time column
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save to CSV
    folder_name = 'reddit_data'
    os.makedirs(folder_name, exist_ok=True)
    normalized_subreddit = normalize_subreddit_name(subreddit)
    csv_filename = os.path.join(folder_name, f"{normalized_subreddit}_reddit_data.csv")
    df.to_csv(csv_filename, sep=',', index=False)
    print(f"Reddit data for {subreddit} saved to {csv_filename}")

    # Save to MongoDB
    subreddit_collection = db[normalized_subreddit]
    data = df.to_dict(orient='records')
    for record in data:
        if subreddit_collection.count_documents({'Link': record['Link']}, limit=1) == 0:
            subreddit_collection.insert_one(record)
            print(f"Data for {subreddit} inserted into MongoDB successfully.")
        else:
            print(f"Data for {subreddit} already exists in MongoDB. Skipping insertion.")

    print(f"Reddit data for {subreddit} added to MongoDB")
