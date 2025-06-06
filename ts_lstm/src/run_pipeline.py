import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go

from fetch_data import download_sp500_data, download_fred_data
from features import create_technical_features
from prepare import prepare_data
from model import build_lstm_model, create_early_stopping, create_model_checkpoint
from eval import calculate_metrics, plot_predictions, backtest_strategy

def main():
    # 1. Download data
    print("Downloading S&P 500 and macro data...")
    sp500_data = download_sp500_data()
    fred_data = download_fred_data(['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS'])

    # 2. Feature engineering
    print("Creating features...")
    features = create_technical_features(sp500_data)
    features = features.join(fred_data, how='left')
    features = features.fillna(method='ffill').dropna()

    # 3. Prepare sequences and scale data
    print("Preparing data for LSTM...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(
        features,
        target_col='returns',
        sequence_length=60,
        target_length=1
    )

    # 4. Build and train model
    print("Building and training LSTM model...")
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=50,
        dropout_rate=0.2
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            create_early_stopping(patience=10),
            create_model_checkpoint('best_model.keras')
        ],
        verbose=1
    )

    # 5. Evaluate model
    print("Evaluating model...")
    model.load_weights('best_model.keras')
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 6. Plot predictions
    print("Plotting predictions...")
    # Create dummy arrays with the same number of features, fill with zeros
    n_features = len(features.columns)
    y_test_full = np.zeros((len(y_test), n_features))
    y_pred_full = np.zeros((len(y_pred), n_features))

    # Find the index of the 'returns' column
    returns_idx = list(features.columns).index('returns')

    # Place your predictions and true values in the correct column
    y_test_full[:, returns_idx] = y_test.flatten()
    y_pred_full[:, returns_idx] = y_pred.flatten()

    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test_full)[:, returns_idx]
    y_pred_inv = scaler.inverse_transform(y_pred_full)[:, returns_idx]

    # Reconstruct price series from returns
    test_start_idx = len(features) - len(y_test_inv)
    start_price = sp500_data['Close'].iloc[test_start_idx - 1]

    actual_prices = [start_price]
    predicted_prices = [start_price]
    for actual_r, pred_r in zip(y_test_inv, y_pred_inv):
        actual_prices.append(actual_prices[-1] * (1 + actual_r))
        predicted_prices.append(predicted_prices[-1] * (1 + pred_r))
    actual_prices = actual_prices[1:]
    predicted_prices = predicted_prices[1:]

    # Plot actual vs predicted prices
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(y=actual_prices, name='Actual Price', line=dict(color='blue')))
    fig_price.add_trace(go.Scatter(y=predicted_prices, name='Predicted Price', line=dict(color='red')))
    fig_price.update_layout(title='Actual vs Predicted Prices', xaxis_title='Time', yaxis_title='Price', showlegend=True)
    fig_price.write_html("price_predictions_plot.html")

    # 7. Backtest strategy
    print("Running backtest...")
    results = backtest_strategy(y_test_inv, y_pred_inv, threshold=0.001)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=results['cumulative_returns'], name='Buy & Hold', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(y=results['cumulative_strategy_returns'], name='Strategy', line=dict(color='red')))
    fig2.update_layout(title='Cumulative Returns', xaxis_title='Time', yaxis_title='Cumulative Return', showlegend=True)
    fig2.write_html("cumulative_returns_plot.html")

if __name__ == "__main__":
    main()