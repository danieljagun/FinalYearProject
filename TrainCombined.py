import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

crypto_folder = 'crypto_data'
crypto_data = pd.concat([pd.read_csv(os.path.join(crypto_folder, file)) for file in os.listdir(crypto_folder)],
                        ignore_index=True)

reddit_folder = 'reddit_data'
reddit_data = pd.concat([pd.read_csv(os.path.join(reddit_folder, file)) for file in os.listdir(reddit_folder)],
                        ignore_index=True)

# Convert 'Timestamp' column to datetime64[ns] and sort by 'Timestamp'
crypto_data['Timestamp'] = pd.to_datetime(crypto_data['Timestamp'])
reddit_data['Timestamp'] = pd.to_datetime(reddit_data['Timestamp'])

crypto_data = crypto_data.sort_values(by='Timestamp')
reddit_data = reddit_data.sort_values(by='Timestamp')

# List of coins
target_coins = ['bitcoin', 'ethereum', 'solana', 'dogecoin', 'cardano', 'avalanche']

for coin in target_coins:
    print(f"Processing {coin}...")
    coin_crypto_data = crypto_data[crypto_data['Coin'] == coin]

    if coin_crypto_data.empty:
        print(f"No cryptocurrency data available for {coin}.")
        continue

    merged_data = pd.merge_asof(coin_crypto_data, reddit_data[['Timestamp', 'Count', 'Sentiment']],
                                on='Timestamp', direction='nearest', tolerance=pd.Timedelta('1h'))

    if merged_data.empty:
        print(f"No merged data available for {coin}. Check alignment of timestamps.")
        continue

    combined_data = merged_data[['Coin', 'Prices', 'MrkCap', 'TolVol', 'Count', 'Sentiment']]
    combined_data = combined_data.dropna()

    if combined_data.empty:
        print(f"No combined data available after dropping NA for {coin}.")
        continue

    combined_data['Price_Change_Percent'] = combined_data.groupby('Coin')['Prices'].pct_change() * 100
    combined_data['Price_Up_or_Down'] = (combined_data['Price_Change_Percent'] > 0).astype(int)

    X_combined = combined_data[['Prices', 'MrkCap', 'TolVol', 'Count', 'Sentiment']]
    y_combined = combined_data['Price_Up_or_Down']

    if X_combined.empty or y_combined.empty:
        print(f"Insufficient data to train model for {coin}.")
        continue

    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    model_combined = RandomForestClassifier()
    model_combined.fit(X_train_combined, y_train_combined)

    y_pred_combined = model_combined.predict(X_test_combined)
    accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)

    print(f'Model Accuracy for {coin} on Test Set: {accuracy_combined}')
    print(f'Classification Report for {coin}:')
    print(classification_report(y_test_combined, y_pred_combined, zero_division=1))
    print(f'Confusion Matrix for {coin}:')
    print(confusion_matrix(y_test_combined, y_pred_combined))

    model_folder_combined = 'combinedModels'
    os.makedirs(model_folder_combined, exist_ok=True)
    model_path_combined = os.path.join(model_folder_combined, f'{coin.lower()}_model.joblib')
    joblib.dump(model_combined, model_path_combined)
    print(f'Model saved to: {model_path_combined}')
