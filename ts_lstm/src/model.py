import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def build_lstm_model(input_shape, units=50, dropout_rate=0.15, learning_rate=0.001):
    """
    Build and compile LSTM model.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout_rate),
        # Dense head
        Dense(32, activation='selu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_model_checkpoint(filepath):
    """
    Create model checkpoint callback.
    
    Args:
        filepath (str): Path to save model weights
        
    Returns:
        tf.keras.callbacks.ModelCheckpoint: Model checkpoint callback
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

def create_early_stopping(monitor='val_loss', patience=10):
    """
    Create early stopping callback.
    
    Args:
        monitor (str): Metric to monitor
        patience (int): Number of epochs to wait before stopping
        
    Returns:
        tf.keras.callbacks.EarlyStopping: Early stopping callback
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )