import pandas as pd
import joblib
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def predict_price_movement(coin_name):
    # Load Crypto Data from MongoDB Atlas
    client = MongoClient("mongodb+srv://danieljagun:Daniel202@cluster.tokbwgs.mongodb.net/?retryWrites=true&w=majority")
    db = client['crypto']
    collection = db[coin_name.lower()]
    cursor = collection.find()
    crypto_data = pd.DataFrame(list(cursor))

    # Load Reddit Data
    reddit_data_path = f'reddit_data/{coin_name.lower()}_reddit_data.csv'
    reddit_data = pd.read_csv(reddit_data_path)

    # Convert 'Timestamp' column to datetime
    crypto_data['Timestamp'] = pd.to_datetime(crypto_data['Timestamp'])
    reddit_data['Timestamp'] = pd.to_datetime(reddit_data['Timestamp'])

    # Merge Crypto and Reddit Data
    merged_data = pd.merge_asof(crypto_data, reddit_data[['Timestamp', 'Count', 'Sentiment']],
                                on='Timestamp', direction='nearest')

    combined_data = merged_data[['Prices', 'MrkCap', 'TolVol', 'Count', 'Sentiment']]

    # Drop rows with missing values
    combined_data = combined_data.dropna()

    # Load Model
    model_path = f'combinedModels/{coin_name.lower()}_model.joblib'
    model = joblib.load(model_path)

    # Features
    X = combined_data[['Prices', 'MrkCap', 'TolVol', 'Count', 'Sentiment']]

    # Predictions
    predictions = model.predict(X)

    # Add predictions to the DataFrame
    combined_data['Price_Movement_Prediction'] = predictions

    return combined_data[['Timestamp', 'Prices', 'MrkCap', 'TolVol', 'Count', 'Sentiment', 'Price_Movement_Prediction']]


# List of target coins
target_coins = ['Bitcoin', 'Ethereum', 'Solana', 'XRP', 'Cardano', 'Dogecoin']

for coin in target_coins:
    predictions_df = predict_price_movement(coin)

    # Print or save predictions as needed
    print(f'Predictions for {coin}:')
    print(predictions_df)
