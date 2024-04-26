import sys
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from statistics import mean, mode, StatisticsError
from io import BytesIO
import base64
from wordcloud import WordCloud

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")

# News API key
news_api_key = '2f373d0c2b6e42cbaa303f67f2f2481b'


def fetch_news(coin_name, api_key):
    url = f"https://newsapi.org/v2/everything?q={coin_name}&apiKey={api_key}&language=en"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        raise Exception(f"Failed to fetch news for {coin_name}: {response.status_code}")


def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        text = (article.get('title', '') or "") + " " + (article.get('description', '') or "")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        sentiment = outputs.logits.softmax(dim=1).argmax().item()
        sentiments.append(sentiment)
    return sentiments


def calculate_overall_sentiment(sentiments):
    if not sentiments:
        return None, "Neutral"  # Default to Neutral if no sentiments

    average_sentiment = mean(sentiments)
    try:
        majority_sentiment = mode(sentiments)
    except StatisticsError:
        majority_sentiment = "No clear majority"  # Handle case with no clear mode

    # Determine sentiment category based on average_sentiment
    if average_sentiment > 1.5:
        sentiment_label = "Positive"
    elif average_sentiment < 1.5:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return sentiment_label, majority_sentiment


def generate_wordcloud(articles):
    text = " ".join(article['title'] for article in articles if article['title'])
    wordcloud = WordCloud(width=800, height=400).generate(text)
    img = BytesIO()
    wordcloud.to_image().save(img, 'PNG')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def run_analysis(coin_name):
    try:
        articles = fetch_news(coin_name, news_api_key)
        sentiments = analyze_sentiment(articles)
        average_sentiment, majority_sentiment = calculate_overall_sentiment(sentiments)
        print(f"Sentiments for {coin_name}: {sentiments}")
        print(f"Average Sentiment: {average_sentiment}, Majority Sentiment: {majority_sentiment}")
        generate_wordcloud(articles)
    except Exception as e:
        print(f"Error during analysis for {coin_name}: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        coin_name = sys.argv[1]
        run_analysis(coin_name)
    else:
        print("Please provide a coin name.")
