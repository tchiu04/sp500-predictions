import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_sequences(data, target_col, sequence_length=60, target_length=1):
    """
    Create sequences for time series prediction.
    
    Args:
        data (pd.DataFrame): Input features
        target_col (str): Name of target column
        sequence_length (int): Length of input sequences
        target_length (int): Length of target sequences
        
    Returns:
        tuple: (X, y) arrays for model training
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - target_length + 1):
        X.append(data.iloc[i:(i + sequence_length)].values)
        y.append(data.iloc[i + sequence_length:i + sequence_length + target_length][target_col].values)
    
    return np.array(X), np.array(y)

def scale_features(train_data, val_data=None, test_data=None):
    """
    Scale features using StandardScaler.
    
    Args:
        train_data (pd.DataFrame): Training data
        val_data (pd.DataFrame): Validation data
        test_data (pd.DataFrame): Test data
        
    Returns:
        tuple: (Scaled train data, Scaled val data, Scaled test data, Scaler)
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    
    val_scaled = None
    if val_data is not None:
        val_scaled = scaler.transform(val_data)
        
    test_scaled = None
    if test_data is not None:
        test_scaled = scaler.transform(test_data)
        
    return train_scaled, val_scaled, test_scaled, scaler

def prepare_data(df, target_col, sequence_length=60, target_length=1, train_split=0.7, val_split=0.15):
    """
    Prepare data for model training.
    
    Args:
        df (pd.DataFrame): Input features
        target_col (str): Name of target column
        sequence_length (int): Length of input sequences
        target_length (int): Length of target sequences
        train_split (float): Training set proportion
        val_split (float): Validation set proportion
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    # Split data
    train_size = int(len(df) * train_split)
    val_size = int(len(df) * val_split)
    
    train_data = df[:train_size]
    val_data = df[train_size:train_size + val_size]
    test_data = df[train_size + val_size:]
    
    # Scale features
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_data, val_data, test_data
    )
    
    # Create sequences
    X_train, y_train = create_sequences(
        pd.DataFrame(train_scaled, columns=df.columns),
        target_col,
        sequence_length,
        target_length
    )
    
    X_val, y_val = create_sequences(
        pd.DataFrame(val_scaled, columns=df.columns),
        target_col,
        sequence_length,
        target_length
    )
    
    X_test, y_test = create_sequences(
        pd.DataFrame(test_scaled, columns=df.columns),
        target_col,
        sequence_length,
        target_length
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler 