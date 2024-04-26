import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

crypto_folder = 'crypto_data'
crypto_data = pd.concat([pd.read_csv(os.path.join(crypto_folder, file)) for file in os.listdir(crypto_folder)],
                        ignore_index=True)

reddit_folder = 'reddit_data'
reddit_data = pd.concat([pd.read_csv(os.path.join(reddit_folder, file)) for file in os.listdir(reddit_folder)],
                        ignore_index=True)

crypto_data['Timestamp'] = pd.to_datetime(crypto_data['Timestamp'])
reddit_data['Timestamp'] = pd.to_datetime(reddit_data['Timestamp'])

crypto_data = crypto_data.sort_values(by='Timestamp')
reddit_data = reddit_data.sort_values(by='Timestamp')

target_coins = ['Bitcoin', 'Ethereum', 'Solana', 'Dogecoin', 'Cardano', 'Avalanche']

for coin in target_coins:
    coin_crypto_data = crypto_data[crypto_data['Coin'].str.lower() == coin.lower()].copy()
    coin_reddit_data = reddit_data[reddit_data['Subreddit'].str.lower() == coin.lower()].copy()

    coin_reddit_data['Coin'] = coin

    merged_data = pd.merge_asof(coin_crypto_data, coin_reddit_data, on='Timestamp', by='Coin', direction='nearest')

    X = merged_data[['MrkCap', 'TolVol', 'Count', 'roberta_pos']]
    y = merged_data['Prices']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model_combined = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model_combined.fit(X_train, y_train)

    y_pred_combined = model_combined.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_combined)
    r2 = r2_score(y_test, y_pred_combined)

    print(f'{coin} Combined Model - MSE: {mse}, R-squared: {r2}')

    # Saving the Combined Model
    model_folder_combined = 'CombinedXGBoostModels'
    os.makedirs(model_folder_combined, exist_ok=True)
    model_path_combined = os.path.join(model_folder_combined, f'{coin.lower()}_combined_xgboost_model.joblib')
    joblib.dump(model_combined, model_path_combined)
    print(f'Model saved to: {model_path_combined}')
