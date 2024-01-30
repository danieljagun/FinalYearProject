from pprint import pprint

import nltk
import pandas as pd
import praw
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer as Senti

user_agent = "Scraper 1.0 by /u/danieljagun"
reddit = praw.Reddit(
    client_id="K-AfR5cf3qvXEXM_MTn7IQ",
    client_secret="JdAlNMb6dPkWCd4bX4PdXCwIpbA5gQ",
    user_agent=user_agent
)


def get_headlines_with_dates(subreddit, category, limit=None):
    headlines = set()
    timestamps = []
    subreddit_obj = reddit.subreddit(subreddit)
    if category == 'hot':
        for submission in subreddit_obj.hot(limit=limit):
            headlines.add(submission.title)
            timestamps.append(submission.created)  # Append the created timestamp
    elif category == 'new':
        for submission in subreddit_obj.new(limit=limit):
            headlines.add(submission.title)
            timestamps.append(submission.created)  # Append the created timestamp
    elif category == 'rising':
        for submission in subreddit_obj.rising(limit=limit):
            headlines.add(submission.title)
            timestamps.append(submission.created)  # Append the created timestamp
    elif category == 'top':
        for submission in subreddit_obj.top(limit=limit):
            headlines.add(submission.title)
            timestamps.append(submission.created)  # Append the created timestamp

    print(f"Number of {category} headlines for {subreddit}: {len(headlines)}")
    return headlines, timestamps


# Fetch headlines with timestamps
hot_headlines, hot_timestamps = get_headlines_with_dates('CryptoCurrency', 'hot', None)
new_headlines, new_timestamps = get_headlines_with_dates('CryptoCurrency', 'new', None)
rising_headlines, rising_timestamps = get_headlines_with_dates('CryptoCurrency', 'rising', None)
top_headlines, top_timestamps = get_headlines_with_dates('CryptoCurrency', 'top', None)
hot_headlines_bitcoin, hot_timestamps_bitcoin = get_headlines_with_dates('Bitcoin', 'hot', None)
new_headlines_bitcoin, new_timestamps_bitcoin = get_headlines_with_dates('Bitcoin', 'new', None)
rising_headlines_bitcoin, rising_timestamps_bitcoin = get_headlines_with_dates('Bitcoin', 'rising', None)
top_headlines_bitcoin, top_timestamps_bitcoin = get_headlines_with_dates('Bitcoin', 'top', None)

# Combine headlines and timestamps
all_headlines = set(hot_headlines | new_headlines | rising_headlines | top_headlines |
                    hot_headlines_bitcoin | new_headlines_bitcoin | rising_headlines_bitcoin | top_headlines_bitcoin)

all_timestamps = hot_timestamps + new_timestamps + rising_timestamps + top_timestamps + \
                 hot_timestamps_bitcoin + new_timestamps_bitcoin + rising_timestamps_bitcoin + top_timestamps_bitcoin


# Check if lengths match
if len(all_headlines) == len(all_timestamps):
    # Create DataFrame with headlines and timestamps
    df = pd.DataFrame({'headline': list(all_headlines), 'timestamp_utc': all_timestamps})
    df.to_csv('combined_titles_with_timestamps.csv', encoding='utf-8', index=False)

# nltk.download('vader_lexicon')
#
# senia = Senti()
# results = []
#
# for line in all_headlines:
#     pol_score = senia.polarity_scores(line)  # auto sentiment class
#     pol_score['headline'] = line
#     results.append(pol_score)
#
# pprint(results[:3], width=100)
#
# df = pd.DataFrame.from_records(results)
# df.head()
#
# df['label'] = 0
# df.loc[df['compound'] > 0.2, 'label'] = 1  # Positive
# df.loc[df['compound'] < -0.2, 'label'] = -1  # Negative
# df.head()
#
# df2 = df[['headline', 'label']]
#
# df2.to_csv('reddit_titles_scored.csv', encoding='utf-8', index=False)
#
# print("Number of Positive, Neutral, and Negative Titles:")
# print(df.label.value_counts())
# print("\n")
#
# print("Positive Titles:\n")
# pprint(list(df[df['label'] == 1].headline)[:5], width=200)
# print("\n")
#
# print("Negative Titles:\n")
# pprint(list(df[df['label'] == -1].headline)[:5], width=200)







# Load and Preprocess Data
# Read your coin and Reddit sentiment datasets into pandas DataFrames.
# Ensure the timestamps are in a common format for alignment.

# Read coingecko and Reddit sentiment datasets
# coingecko_df = pd.read_csv('coingecko_top_10.csv')
# reddit_df = pd.read_csv('reddit_titles_scored.csv')
#
# # Display the date formats for both datasets
# print("Coingecko 'Item' column format:", coingecko_df['Item'].dtype)
# print("Reddit 'Date' column format:", reddit_df['Date'].dtype)
#
# # Display unique dates from both datasets
# print("\nUnique dates in coingecko dataset:")
# print(coingecko_df['Item'].unique())
#
# print("\nUnique dates in Reddit dataset:")
# print(reddit_df['Date'].unique())
#
# # Merge based on the 'Item' and 'Date' columns
# merged_data = pd.merge(coingecko_df, reddit_df, left_on='Item', right_on='Date', how='inner')
#
# # Save the merged data to a CSV file
# merged_data.to_csv('merged_data.csv', index=False)

# market_df = pd.read_csv("coingecko_top_10.csv")
# reddit_df = pd.read_csv("reddit_titles_scored.csv")
#
# # Check for missing or NaN values in market_df
# print("Missing or NaN values in market_df:")
# print(market_df.isna().sum())
#
# # Check for missing or NaN values in reddit_df
# print("\nMissing or NaN values in reddit_df:")
# print(reddit_df.isna().sum())
#
# # Convert the 'Item' column to datetime format
# market_df['Item'] = pd.to_datetime(market_df['Item'])
#
# # Merge the DataFrames
# merged_df = pd.merge(reddit_df, market_df, left_on='headline', right_on='Coin', how='inner')
#
# merged_df.to_csv('merged_data.csv', index=False)
#
# # Calculate the correlation
# correlation_matrix = merged_df[['Prices', 'label']].corr()
#
# # Visualize the correlation matrix
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.show()
