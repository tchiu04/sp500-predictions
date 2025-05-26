# S&P 500 Prediction using LSTM

This project implements a time-series forecasting model using LSTM to predict short-term S&P 500 returns. The model uses price, volume, and macroeconomic data to make predictions.

## Project Structure

```
ts_lstm/
├── data/
│   ├── raw/          # Raw data files
│   └── processed/    # Processed data files
├── notebooks/        # Jupyter notebooks
├── src/             # Source code
│   ├── fetch_data.py    # Data downloading
│   ├── features.py      # Feature engineering
│   ├── prepare.py       # Data preparation
│   ├── model.py         # LSTM model
│   ├── train.py         # Training script
│   └── eval.py          # Evaluation functions
├── Dockerfile       # Docker configuration
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/sp500_prediction.ipynb`

### Training the Model

1. Run the training script:
```bash
python src/train.py
```

### Building and Running with Docker

1. Build the Docker image:
```bash
docker build -t ts_lstm .
```

2. Run the container:
```bash
docker run ts_lstm
```

## Features

- Downloads S&P 500 and FRED macroeconomic data
- Implements technical indicators (EMA, MACD, momentum)
- Uses PCA for feature dimensionality reduction
- Implements walk-forward validation for time series
- Includes SHAP value analysis for model interpretability
- Provides backtesting functionality

## Model Architecture

The LSTM model consists of:
- Two LSTM layers with dropout
- Dense output layer
- MSE loss function
- Adam optimizer

## Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Directional Accuracy
- Cumulative Returns (Backtest)

## License

MIT 