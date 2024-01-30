import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_crypto_model(crypto_name):
    # Loading crypto price data
    crypto_data_path = f'crypto_data/{crypto_name.lower()}.csv'
    crypto_data = pd.read_csv(crypto_data_path)

    # Defining a Target Variable
    crypto_data['Price_Change_Percent'] = crypto_data.groupby('Coin')['Prices'].pct_change() * 100
    crypto_data['Price_Up_or_Down'] = (crypto_data['Price_Change_Percent'] > 0).astype(int)
    crypto_data = crypto_data.dropna()

    # Feature Engineering
    X = crypto_data[['Prices', 'MrkCap', 'TolVol']]
    y = crypto_data['Price_Up_or_Down']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training a Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Checking Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy on Test Set: {accuracy}')

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Saving Model
    model_folder = 'cryptoModels'
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'{crypto_name.lower()}_crypto_model.joblib')
    joblib.dump(model, model_path)
    print(f'Model saved to: {model_path}')


train_crypto_model('Bitcoin')
train_crypto_model('Ethereum')
train_crypto_model('Solana')
train_crypto_model('XRP')
train_crypto_model('Cardano')
train_crypto_model('Dogecoin')
