import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def calculate_ema(series, span):
    """
    Calculate Exponential Moving Average.
    
    Args:
        series (pd.Series): Price series
        span (int): EMA period
        
    Returns:
        pd.Series: EMA values
    """
    return series.ewm(span=span, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        series (pd.Series): Price series
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period
        
    Returns:
        tuple: (MACD line, Signal line, MACD histogram)
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_momentum(series, period=14):
    """
    Calculate momentum indicator.
    
    Args:
        series (pd.Series): Price series
        period (int): Lookback period
        
    Returns:
        pd.Series: Momentum values
    """
    return series - series.shift(period)

def apply_pca(df, n_components=0.95):
    """
    Apply PCA to reduce dimensionality.
    
    Args:
        df (pd.DataFrame): Input features
        n_components (float): Explained variance ratio threshold
        
    Returns:
        tuple: (Transformed data, PCA object)
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(df)
    return transformed, pca

def create_technical_features(df):
    """
    Create technical indicators from price data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'])
    
    # Volume-based features
    features['volume_ma'] = df['Volume'].rolling(window=20).mean()
    features['volume_std'] = df['Volume'].rolling(window=20).std()
    
    # Technical indicators
    features['ema_20'] = calculate_ema(df['Close'], 20)
    features['ema_50'] = calculate_ema(df['Close'], 50)
    
    macd_line, signal_line, histogram = calculate_macd(df['Close'])
    features['macd'] = macd_line
    features['macd_signal'] = signal_line
    features['macd_hist'] = histogram
    
    features['momentum'] = calculate_momentum(df['Close'])
    
    return features.dropna() 