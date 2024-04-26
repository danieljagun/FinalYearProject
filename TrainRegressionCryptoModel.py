import os

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def feature_engineering(crypto_data):
    crypto_data = crypto_data.copy()
    crypto_data['Price_Change_Percent'] = crypto_data.groupby('Coin')['Prices'].pct_change() * 100
    crypto_data['Previous_Price'] = crypto_data.groupby('Coin')['Prices'].shift(1)
    crypto_data['Rolling_Mean'] = crypto_data.groupby('Coin')['Prices'].rolling(window=5).mean().reset_index(0,
                                                                                                             drop=True)
    crypto_data['Price_TolVol_Interact'] = crypto_data['Prices'] * crypto_data['TolVol']
    crypto_data['Day_of_Week'] = pd.to_datetime(crypto_data['Timestamp']).dt.dayofweek
    return crypto_data.dropna()


def train_crypto_model(crypto_data):
    X = crypto_data[['MrkCap', 'TolVol', 'Previous_Price', 'Rolling_Mean', 'Price_TolVol_Interact', 'Day_of_Week']]
    y = crypto_data['Prices']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, model


def save_model(model, crypto_name):
    model_folder = 'RegressionCryptoModels'
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'{crypto_name.lower()}_linear_model.joblib')
    joblib.dump(model, model_path)
    print(f'Model saved to: {model_path}')


cryptocurrencies = ['Bitcoin', 'Ethereum', 'Solana', 'Dogecoin', 'Cardano', 'Avalanche']
for crypto in cryptocurrencies:
    try:
        crypto_data_path = f'crypto_data/{crypto.lower()}.csv'
        crypto_data = pd.read_csv(crypto_data_path)
        crypto_data = feature_engineering(crypto_data)
        mse, r2, model = train_crypto_model(crypto_data)
        print(f'{crypto} - Model MSE: {mse}, R-squared: {r2}')
        save_model(model, crypto)
    except Exception as e:
        print(f'Failed to train model for {crypto}: {str(e)}')
