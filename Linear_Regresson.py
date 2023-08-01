#!/usr/bin/env python
# coding: utf-8

# In[27]:


import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from datetime import timedelta
import boto3
import os
import json
import math
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')




with open('/Users/cedrix/Documents/aws.json', 'r') as f:
    credentials = json.load(f)

# Set environment variables
os.environ['AWS_ACCESS_KEY_ID'] = credentials ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = credentials ['AWS_SECRET_ACCESS_KEY']

# AWS S3 bucket
bucket = 'raw-stock-price'

# Load data from S3
  
def load_data_from_s3(file_name):
    s3 = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])
    obj = s3.get_object(Bucket=bucket, Key=file_name)
    df = pd.read_csv(obj['Body'])
    return df

# Function to list all files in a specific S3 bucket folder
def list_files_in_s3_bucket(bucket_name, prefix):
    s3 = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])
    response = s3.list_objects(Bucket=bucket, Prefix=prefix)

    # Get a list of all the file names
    files = [item['Key'] for item in response['Contents']]

    # Extract the stock symbol from each file name
    stock_symbols = [file.split('/')[-1].split('_')[0] for file in files]

    return stock_symbols

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.dropna(inplace=True)  # Drop rows with missing data
    return df

def add_feature(df, feature, window):
    if feature == 'MA':
        close_col = df['adj_close']
        df['MA'] = close_col.rolling(window=window).mean()
    if feature == 'EMA':
        close_col = df['adj_close']
        df['EMA'] = close_col.ewm(span=window, adjust=False).mean()
    if feature == 'SO':
        high14 = df['high'].rolling(window).max()
        low14 = df['low'].rolling(window).min()
        df['%K'] = (df['close'] - low14) * 100 / (high14 - low14)
        df['%D'] = df['%K'].rolling(3).mean()
    return df

 
def train_model(df, future_days, test_size):
    try:
    
        # Apply shift operation
        df['Prediction'] = df['adj_close'].shift(-future_days)

        df_copy = df.copy()

        # Create X_predict using the shifted copy
        X_predict = np.array(df_copy.drop(['Prediction'], 1))[-future_days:]
        X_predict = np.array(df.drop(['Prediction'], 1))[-future_days:]
        # print(X_predict)  

        X = np.array(df.drop(['Prediction'], axis=1))
        X = X[:-future_days]
        y = np.array(df['Prediction'])
        y = y[:-future_days]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model = LinearRegression()

        with mlflow.start_run():
            mlflow.log_param("future_days", future_days)
            mlflow.log_param("test_size", test_size)

            model.fit(X_train, y_train)

            # Log model
            mlflow.sklearn.log_model(model, "linear_regression")

            # Log metrics: RMSE, MSE and MAPE
            rmse = math.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            mse = mean_squared_error(y_test, model.predict(X_test))
            mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mape", mape)



        model.fit(X_train, y_train)

        return model, X_train, X_test, y_train, y_test, X_predict


        # Generate prediction
        linear_model_predict_prediction = model.predict(X_predict)
        linear_model_real_prediction = model.predict(np.array(df.drop(['Prediction'], 1)))

        return model, X_train, X_test, y_train, y_test, linear_model_real_prediction, linear_model_predict_prediction

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mlflow.end_run()


def evaluate_model(model, X_test, y_test, metric):
    predictions = model.predict(X_test)
    if metric == 'rmse':
        return math.sqrt(mean_squared_error(y_test, predictions))
    elif metric == 'mse':
        return mean_squared_error(y_test, predictions)
    elif metric == 'mape':
        return mean_absolute_percentage_error(y_test, predictions)
    else:
        return None

def plot_results(df, linear_model_real_prediction, linear_model_predict_prediction, display_at, future_days, alpha):
    predicted_dates = [df.index[-1] + timedelta(days=x) for x in range(1, future_days+1)]
    fig, ax = plt.subplots(figsize=(40, 20))

    # Change the background color to black
    plt.rcParams['figure.facecolor'] = 'black'
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.plot(df.index[display_at:], linear_model_real_prediction[display_at:], label='Linear Prediction', color='magenta', alpha=alpha, linewidth=5.0)
    ax.plot(predicted_dates, linear_model_predict_prediction, label='Forecast', color='aqua', alpha=alpha, linewidth=5.0)
    ax.plot(df.index[display_at:], df['adj_close'][display_at:], label='Actual', color='lightgreen', linewidth=5.0)

    # Format the x-axis dates
    date_format = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)

    plt.legend(prop={'size': 35})  # Increase the size of the legend
    plt.xticks(fontsize=30)  # Increase x-axis font size
    plt.yticks(fontsize=30)  # Increase y-axis font size
    plt.show()


def run_model(file_name, ma_window=5, ema_window=5, so_window=5, features=['MA','close', 'EMA', 'SO'], test_size=0.5, future_days=30, rmse=True, mse=True, mape=True, display_at=0, alpha=0.5):
    
    try:
        logging.info("Running model...")  
        # Load data from S3
        df = load_data_from_s3(file_name)
        logging.info(f"loaded data: {df}")  


        

        # Preprocess data
        df = preprocess_data(df)
        logging.info(f"Preprocessed data: {df}")  

        # Define feature windows
        feature_windows = {
            'MA': ma_window,
            'EMA': ema_window,
            'SO': so_window
        }

        # Add features to the data
        for feature in features:
            if feature in feature_windows:
                df = add_feature(df, feature, feature_windows[feature])

        model, X_train, X_test, y_train, y_test, X_predict = train_model(df, future_days, test_size)
        logging.info(f"Trained model: {model}")  

        # Train model and evaluate
        model, X_train, X_test, y_train, y_test, X_predict = train_model(df, future_days, test_size)
        evaluations = {}
        if rmse:
            evaluations['rmse'] = evaluate_model(model, X_test, y_test, 'rmse')
        if mse:
            evaluations['mse'] = evaluate_model(model, X_test, y_test, 'mse')
        if mape:
            evaluations['mape'] = evaluate_model(model, X_test, y_test, 'mape')

        loffinf.info(f"Evaluations: {evaluations}")



        

        # Generate prediction
        linear_model_real_prediction = model.predict(np.array(df.drop(['Prediction'], 1)))
        linear_model_predict_prediction = model.predict(X_predict)

        # Plot
        plot_results(df, linear_model_real_prediction, linear_model_predict_prediction, display_at, future_days, alpha)

    

        return model, evaluations, df
        

    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, None, None  # return None for each expected return value
    finally:
        mlflow.end_run()



# usage:
model, evaluations, df = run_model('yhoofinance-daily-historical-data/TSLA_daily_data.csv', ma_window=5, ema_window=5, so_window=5, features=['adj_close'], test_size=0.2, future_days=30, rmse=True, mse=True, mape=True, display_at=0, alpha=0.5)


# In[34]:


def main():

  
    
    
    st.title('Linear Regression Stock Price Prediction')
     
    st.sidebar.markdown('# Parameters')

    # Get a list of all the stock symbols in the 'yhoofinance-daily-historical-data/' folder
    stock_symbols = list_files_in_s3_bucket('raw-stock-price', 'yhoofinance-daily-historical-data/')

    # Use this list to populate the dropdown menu
    stock_symbol = st.sidebar.selectbox('Stocks', stock_symbols)

    # Construct the file name from the selected stock symbol
    file_name = f'yhoofinance-daily-historical-data/{stock_symbol}_daily_data.csv'
    ma_window = st.sidebar.slider('Moving Avg. -- Window Size', 1, 100, 50)
    ema_window = st.sidebar.slider('Exponential Moving Avg. -- Window Size', 1, 100, 50)
    so_window = st.sidebar.slider('Stochastic Oscillator -- Window Size', 1, 100, 50)
    test_size = st.sidebar.slider('Test Set Size', 0.1, 0.9, 0.2)
    future_days = st.sidebar.slider('Days to Forecast', 1, 50, 30)
    display_at = st.sidebar.slider('Display From Day', 0, 365, 0)
    alpha = st.sidebar.slider('Alpha', 0.1, 1.0, 0.5)

    features = st.sidebar.multiselect('Features', options=['MA', 'EMA', 'SO', 'adj_close'], default=['adj_close'])

    metrics = st.sidebar.multiselect('Evaluation Metrics', options=['RMSE', 'MSE', 'MAPE'], default=['RMSE', 'MSE', 'MAPE'])

    rmse = 'RMSE' in metrics
    mse = 'MSE' in metrics
    mape = 'MAPE' in metrics

    if st.sidebar.button('Train Model'):
    
        st.markdown('## Training Model...')

        model, evaluations, df = run_model(
            file_name=file_name,
            ma_window=ma_window,
            ema_window=ema_window,
            so_window=so_window,
            features=features,
            test_size=test_size,
            future_days=future_days,
            rmse=rmse,
            mse=mse,
            mape=mape,
            display_at=display_at,
            alpha=alpha
        )

        result = run_model(
        file_name=file_name,
        ma_window=ma_window,
        ema_window=ema_window,
        so_window=so_window,
        features=features,
        test_size=test_size,
        future_days=future_days,
        rmse=rmse,
        mse=mse,
        mape=mape,
        display_at=display_at,
        alpha=alpha
        )

        # Display evaluation metrics in multiple columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("RMSE")
            st.write(evaluations['rmse'])

        with col2:
            st.header("MSE")
            st.write(evaluations['mse'])

        with col3:
            st.header("MAPE")
            st.write(evaluations['mape'])

        st.markdown('## Forecast Plot')
        st.pyplot()

        if result is not None:
             st.markdown('')
        else:
            st.markdown('## An error occurred during model training')




if __name__ == '__main__':
    main()

