import logging
import os
import subprocess
import re
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from RegressionPrediction import CRYPTOCURRENCIES, make_price_prediction
from pymongo import MongoClient
import requests

from NewsSentiment import run_analysis, fetch_news, analyze_sentiment, generate_wordcloud, news_api_key, \
    calculate_overall_sentiment

os.environ["TOKENIZERS_PARALLELISM"] = "true"

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# MongoDB connection setup
mongo_uri = "mongodb+srv://danieljagun:Daniel202@cluster.tokbwgs.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)
db = client['predictions']
predictions_collection = db['predictions_data']


def fetch_actual_price(coin_id):
    coin_id = coin_id.lower()
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['market_data']['current_price']['usd']
    except requests.RequestException as e:
        logging.error(f"Failed to fetch current price for {coin_id}: {str(e)}")
        return None


@app.route('/')
def hello_world():
    return jsonify({'message': 'Hello, Test!'})


@app.route('/predict', methods=['POST'])
def predict():
    coin_name = request.json.get('coin_name', None)
    if coin_name is None:
        return jsonify({'success': False, 'error': 'No coin name provided'})

    try:
        result = subprocess.check_output(['python', 'TrainModelPrediction.py', coin_name]).decode('utf-8')

        match = re.search(r"The predicted price movement for \w+ is: (Up|Down)", result)
        if match:
            prediction_outcome = match.group(1)
        else:
            prediction_outcome = "Unknown"

        # Store prediction in MongoDB
        prediction_data = {
            'coin_name': coin_name.lower(),
            'predicted_movement': prediction_outcome,
            'full_result': result,
            'timestamp': datetime.utcnow()
        }
        predictions_collection.insert_one(prediction_data)
        return jsonify({'success': True, 'output': result})
    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/pricePredict', methods=['POST'])
def price_predict():
    coin_name = request.json.get('coin_name', None)
    if coin_name is None:
        return jsonify({'success': False, 'error': 'No coin name provided'})

    if coin_name not in CRYPTOCURRENCIES:
        return jsonify({'success': False, 'error': 'Invalid coin name'})

    try:
        actual_price = fetch_actual_price(coin_name)
        predicted_price = make_price_prediction(coin_name)
        if predicted_price is not None and actual_price is not None:
            prediction_data = {
                'coin_name': coin_name.lower(),
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'timestamp': datetime.utcnow()
            }
            predictions_collection.insert_one(prediction_data)

            return jsonify({'success': True, 'predicted_price': predicted_price, 'actual_price': actual_price})
        else:
            return jsonify({'success': False, 'error': 'Prediction or price fetch failed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/predictions/<coin_name>', methods=['GET'])
def get_predictions(coin_name):
    try:
        predictions = predictions_collection.find({'coin_name': coin_name})
        predictions_list = list(predictions)
        for prediction in predictions_list:
            prediction['_id'] = str(prediction['_id'])  #
        return jsonify({'success': True, 'data': predictions_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/news_sentiment', methods=['POST'])
def news_sentiment():
    coin_name = request.json.get('coin_name')
    if not coin_name:
        return jsonify({'success': False, 'error': 'No coin name provided'})

    try:
        articles = fetch_news(coin_name, news_api_key)
        sentiments = analyze_sentiment(articles)
        sentiment_label, majority_sentiment = calculate_overall_sentiment(sentiments)
        word_cloud = generate_wordcloud(articles)

        return jsonify({
            'success': True,
            'overall_sentiment': sentiment_label,
            'majority_sentiment': majority_sentiment,
            'word_cloud_url': word_cloud
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
