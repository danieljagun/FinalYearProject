import os

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def feature_engineering(crypto_data):
    crypto_data = crypto_data.copy()  # Creating copy to avoid SettingWithCopyWarning

    crypto_data['Price_Change_Percent'] = crypto_data.groupby('Coin')['Prices'].pct_change() * 100
    crypto_data['Price_Up_or_Down'] = (crypto_data['Price_Change_Percent'] > 0).astype(int)
    crypto_data = crypto_data.dropna()

    # Lag Features
    crypto_data['Previous_Price'] = crypto_data.groupby('Coin')['Prices'].shift(1)

    # Rolling Statistics
    crypto_data['Rolling_Mean'] = crypto_data.groupby('Coin')['Prices'].rolling(window=5).mean().reset_index(0,
                                                                                                             drop=True)
    # Interaction Terms
    crypto_data['Price_TolVol_Interact'] = crypto_data['Prices'] * crypto_data['TolVol']

    # Time Features
    crypto_data['Day_of_Week'] = pd.to_datetime(crypto_data['Timestamp']).dt.dayofweek

    return crypto_data


def train_crypto_model(crypto_name):
    # Loading crypto price data
    crypto_data_path = f'crypto_data/{crypto_name.lower()}.csv'
    crypto_data = pd.read_csv(crypto_data_path)

    # Feature Engineering
    crypto_data = feature_engineering(crypto_data)

    # Drop rows with NaN values
    crypto_data = crypto_data.dropna()

    # Define Features and Target
    X = crypto_data[
        ['Prices', 'MrkCap', 'TolVol', 'Previous_Price', 'Rolling_Mean', 'Price_TolVol_Interact', 'Day_of_Week']]
    y = crypto_data['Price_Up_or_Down']

    # Check if more than one class is present in 'y' for oversampling
    if len(y.unique()) > 1:
        # Apply oversampling to handle class imbalance
        oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X, y)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,
                                                            stratify=y_resampled)
    else:
        # No oversampling needed, proceed with regular train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training a Model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)  # Adjust hyperparameters
    model.fit(X_train, y_train)

    # Checking Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy on Test Set: {accuracy}')

    print('Classification Report:')
    print(classification_report(y_test, y_pred, zero_division=1))

    # Saving Model
    model_folder = 'cryptoModels'
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'{crypto_name.lower()}_crypto_model.joblib')
    joblib.dump(model, model_path)
    print(f'Model saved to: {model_path}')


# Train Models
train_crypto_model('Bitcoin')
train_crypto_model('Ethereum')
train_crypto_model('Solana')
train_crypto_model('Dogecoin')
train_crypto_model('Cardano')
train_crypto_model('Avalanche')
