import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timedelta
import os
import yfinance as yf

def download_sp500_data(start_date='2010-01-01', end_date=None):
    """
    Download S&P 500 historical data using yfinance.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    df = yf.download('^GSPC', start=start_date, end=end_date)
    return df

def download_fred_data(series_list, start_date='2010-01-01', end_date=None):
    """
    Download FRED macroeconomic data.
    
    Args:
        series_list (list): List of FRED series IDs
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
        
    Returns:
        pd.DataFrame: DataFrame with FRED data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    df = pdr.data.get_data_fred(series_list, start=start_date, end=end_date)
    return df

def save_data(df, filename, data_dir='../data/raw'):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Output filename
        data_dir (str): Directory to save the file
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath)
    print(f"Saved data to {filepath}")

def main():
    # Download S&P 500 data
    sp500_data = download_sp500_data()
    save_data(sp500_data, 'sp500_daily.csv')
    
    # Download FRED data
    fred_series = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS']  # Example series
    fred_data = download_fred_data(fred_series)
    save_data(fred_data, 'fred_macro.csv')

if __name__ == "__main__":
    main() 