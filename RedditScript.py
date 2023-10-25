from pprint import pprint

import nltk
import pandas as pd
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as Senti

user_agent = "Scraper 1.0 by /u/danieljagun"
reddit = praw.Reddit(
    client_id="K-AfR5cf3qvXEXM_MTn7IQ",
    client_secret="JdAlNMb6dPkWCd4bX4PdXCwIpbA5gQ",
    user_agent=user_agent
)


def get_headlines(subreddit, category, limit=None):
    headlines = set()
    subreddit_obj = reddit.subreddit(subreddit)
    if category == 'hot':
        for submission in subreddit_obj.hot(limit=limit):
            headlines.add(submission.title)
    elif category == 'new':
        for submission in subreddit_obj.new(limit=limit):
            headlines.add(submission.title)
    elif category == 'rising':
        for submission in subreddit_obj.rising(limit=limit):
            headlines.add(submission.title)
    elif category == 'top':
        for submission in subreddit_obj.top(limit=limit):
            headlines.add(submission.title)
    print(f"Number of {category} headlines for {subreddit}: {len(headlines)}")
    return headlines


# hot - r/CryptoCurrency
hot_headlines = get_headlines('CryptoCurrency', 'hot', None)

# new - r/CryptoCurrency
new_headlines = get_headlines('CryptoCurrency', 'new', None)

# rising - r/CryptoCurrency
rising_headlines = get_headlines('CryptoCurrency', 'rising', None)

# top - r/CryptoCurrency
top_headlines = get_headlines('CryptoCurrency', 'top', None)

# hot - r/Bitcoin
hot_headlines_bitcoin = get_headlines('Bitcoin', 'hot', None)

# new - r/Bitcoin
new_headlines_bitcoin = get_headlines('Bitcoin', 'new', None)

# rising - r/Bitcoin
rising_headlines_bitcoin = get_headlines('Bitcoin', 'rising', None)

# top - r/Bitcoin
top_headlines_bitcoin = get_headlines('Bitcoin', 'top', None)

all_headlines = hot_headlines.union(new_headlines, rising_headlines, top_headlines, hot_headlines_bitcoin,
                                    new_headlines_bitcoin, rising_headlines_bitcoin, top_headlines_bitcoin)

df = pd.DataFrame(all_headlines)
df.head()
df.to_csv('combined_titles.csv', header=False, encoding='utf-8', index=False)

nltk.download('vader_lexicon')

senia = Senti()
results = []

for line in all_headlines:
    pol_score = senia.polarity_scores(line)  # auto sentiment class
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
df.head()

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1  # Positive
df.loc[df['compound'] < -0.2, 'label'] = -1  # Negative
df.head()

df2 = df[['headline', 'label']]

df2.to_csv('reddit_titles_scored.csv', encoding='utf-8', index=False)

print("Number of Positive, Neutral, and Negative Titles:")
print(df.label.value_counts())
print("\n")

print("Positive Titles:\n")
pprint(list(df[df['label'] == 1].headline)[:5], width=200)
print("\n")

print("Negative Titles:\n")
pprint(list(df[df['label'] == -1].headline)[:5], width=200)
