import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, accuracy_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    true_direction = np.sign(np.diff(y_true.flatten()))
    pred_direction = np.sign(np.diff(y_pred.flatten()))
    directional_accuracy = accuracy_score(true_direction, pred_direction)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'directional_accuracy': directional_accuracy
    }

def plot_predictions(y_true, y_pred, dates=None):
    """
    Plot true vs predicted values using Plotly.
    
    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values
        dates (pd.DatetimeIndex): Dates for x-axis
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    fig = go.Figure()
    
    if dates is not None:
        x = dates
    else:
        x = np.arange(len(y_true))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y_true.flatten(),
        name='True',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y_pred.flatten(),
        name='Predicted',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='True vs Predicted Values',
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True
    )
    
    return fig

def calculate_shap_values(model, X_background, X_explain):
    """
    Calculate SHAP values for model predictions.
    
    Args:
        model (tf.keras.Model): Trained model
        X_background (np.array): Background data for SHAP
        X_explain (np.array): Data to explain
        
    Returns:
        tuple: (SHAP values, feature names)
    """
    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(X_explain)
    
    return shap_values

def plot_shap_summary(shap_values, feature_names):
    """
    Plot SHAP summary plot.
    
    Args:
        shap_values (np.array): SHAP values
        feature_names (list): List of feature names
        
    Returns:
        matplotlib.figure.Figure: SHAP summary plot
    """
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        plot_type="bar"
    )

def backtest_strategy(y_true, y_pred, threshold=0.0):
    """
    Simple backtest of trading strategy.
    
    Args:
        y_true (np.array): True returns
        y_pred (np.array): Predicted returns
        threshold (float): Trading threshold
        
    Returns:
        pd.DataFrame: Backtest results
    """
    # Ensure 1D arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    signals = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    returns = y_true
    strategy_returns = signals * returns
    
    results = pd.DataFrame({
        'true_returns': returns,
        'predicted_returns': y_pred,
        'signals': signals,
        'strategy_returns': strategy_returns
    })
    
    results['cumulative_returns'] = (1 + results['true_returns']).cumprod()
    results['cumulative_strategy_returns'] = (1 + results['strategy_returns']).cumprod()
    
    return results