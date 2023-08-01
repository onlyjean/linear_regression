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



with open('/Users/cedrix/Documents/aws.json', 'r') as f:
    credentials = json.load(f)

# Set environment variables
os.environ['AWS_ACCESS_KEY_ID'] = credentials ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = credentials ['AWS_SECRET_ACCESS_KEY']

# AWS S3 bucket
bucket = 'raw-stock-price'

# Load data from S3
  
def load_data_from_s3(stock_name):
    file_name = f'yhoofinance-daily-historical-data/{stock_name}_daily_data.csv'
    s3 = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])
    obj = s3.get_object(Bucket=bucket, Key=file_name)
    df = pd.read_csv(obj['Body'])
    return df


def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    #df.dropna(inplace=True)  # Drop rows with missing data
    return df

def add_feature(df, feature, window):
    if feature == 'MA':
        close_col = df['adj_close']
        df['MA'] = close_col.rolling(window=window).mean()
        df['MA'].fillna((df['MA'].mean()), inplace=True)
    if feature == 'EMA':
        close_col = df['adj_close']
        df['EMA'] = close_col.ewm(span=window, adjust=False).mean()
    if feature == 'STO':
        high14 = df['high'].rolling(window).max()
        low14 = df['low'].rolling(window).min()
        df['%K'] = (df['close'] - low14) * 100 / (high14 - low14)
        df['%K'].fillna((df['%K'].mean()), inplace=True)
        df['%D'] = df['%K'].rolling(3).mean()
        df['%D'].fillna((df['%D'].mean()), inplace=True)
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

    # Plot MA, EMA, and STO if they exist in the dataframe
    if 'MA' in df.columns:
        ax.plot(df.index[display_at:], df['MA'][display_at:], label='MA', color='white', alpha=alpha, linewidth=10.0)
    if 'EMA' in df.columns:
        ax.plot(df.index[display_at:], df['EMA'][display_at:], label='EMA', color='red', alpha=alpha, linewidth=10.0)
    if '%K' in df.columns:
        ax.plot(df.index[display_at:], df['%K'][display_at:], label='%K', color='yellow', alpha=alpha, linewidth=10.0)
    if '%D' in df.columns:
        ax.plot(df.index[display_at:], df['%D'][display_at:], label='%D', color='blue', alpha=alpha, linewidth=10.0)

    
        

    # Format the x-axis dates
    date_format = DateFormatter("%Y-%m-%d")


    # Format the x-axis dates
    date_format = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)

    plt.legend(prop={'size': 35})  # Increase the size of the legend
    plt.xticks(fontsize=30)  # Increase x-axis font size
    plt.yticks(fontsize=30)  # Increase y-axis font size
    plt.show()


def run_model(stock_name, ma_window=5, ema_window=5, sto_window=5, features=['MA','close', 'EMA', 'STO'], test_size=0.5, future_days=30, rmse=True, mse=True, mape=True, display_at=0, alpha=0.5):
    model, evaluations, df = None, None, None

    try:
        st.write("Running model...")  
        # Load data from S3
        df = load_data_from_s3(stock_name)
        #st.write(f"loaded data: {df}")
        

        st.header("Data")
        # Preprocess data
        df = preprocess_data(df)
        #st.write(df.shape)
        st.write(df)
        

        # Define feature windows
        feature_windows = {
            'MA': ma_window,
            'EMA': ema_window,
            'STO': sto_window
        }

        # Add features to the data
        for feature in features:
            if feature in feature_windows:
                df = add_feature(df, feature, feature_windows[feature])
            #st.write(df)

        model, X_train, X_test, y_train, y_test, X_predict = train_model(df, future_days, test_size)
        st.write(f"Trained model: {model}")  

        # Train model and evaluate
        evaluations = {}
        if rmse:
            evaluations['rmse'] = evaluate_model(model, X_test, y_test, 'rmse')
        if mse:
            evaluations['mse'] = evaluate_model(model, X_test, y_test, 'mse')
        if mape:
            evaluations['mape'] = evaluate_model(model, X_test, y_test, 'mape')

        #st.write(f"Evaluations: {evaluations}")

        # Generate prediction
        linear_model_real_prediction = model.predict(np.array(df.drop(['Prediction'], 1)))
        linear_model_predict_prediction = model.predict(X_predict)

        # Plot
        plot_results(df, linear_model_real_prediction, linear_model_predict_prediction, display_at, future_days, alpha)


    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Always return a tuple of three elements
    return model, evaluations, df

# In[34]:


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #st.sidebar.image('/Users/cedrix/Downloads/futurstox-high-resolution-logo-white-on-transparent-background.png', width=200)


    with open("/Users/cedrix/Downloads/futurstox-high-resolution-logo-white-on-transparent-background.png", "rb") as img_file:
        logo_img = base64.b64encode(img_file.read()).decode("utf-8")
        
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src='data:image/png;base64,{logo_img}' style='height: 200px;'>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title('Linear Regression Stock Price Prediction')

    st.sidebar.markdown('# Parameters')
    stock_name = st.sidebar.selectbox("Select a stock", ("AAPL", "GOOGL", "MSFT", "AMZN","TSLA","META", "NFLX", "NVDA"), help="Select the stock for which you want to predict the price.")
    future_days = st.sidebar.slider('Days to Forecast', 1, 50, 30, help="Select the number of days in the future for which you want to predict the stock price.")
    display_at = st.sidebar.slider('Display From Day', 0, 365, 0, help="Select the starting day from which you want to display the stock price.")
    metrics = st.sidebar.multiselect('Evaluation Metrics', options=['RMSE', 'MSE', 'MAPE'], default=['RMSE', 'MSE', 'MAPE'], help="Select the metrics you want to use to evaluate the model's performance.")

    rmse = 'RMSE' in metrics
    mse = 'MSE' in metrics
    mape = 'MAPE' in metrics


    # Advanced Settings
    advanced_settings = st.sidebar.checkbox('Advanced Settings')
    if advanced_settings:
        ma_window = st.sidebar.slider('Moving Avg. -- Window Size', 1, 100, 50, help="Select the window size for the moving average.")
        ema_window = st.sidebar.slider('Exponential Moving Avg. -- Window Size', 1, 100, 50, help="Select the window size for the exponential moving average.")
        sto_window = st.sidebar.slider('Stochastic Oscillator -- Window Size', 1, 100, 50, help="Select the window size for the stochastic oscillator.")
        test_size = st.sidebar.slider('Test Set Size', 0.1, 0.9, 0.2, help="Select the size of the test set.")
        alpha = st.sidebar.slider('Alpha', 0.1, 1.0, 0.5, help="Select the alpha value for the model.")
        features = st.sidebar.multiselect('Features', options=['MA', 'EMA', 'STO', 'adj_close'], default=['adj_close'], help="Select the features you want to include in the model.")
    else:
        ma_window = 50
        ema_window = 50
        sto_window = 50
        test_size = 0.2
        alpha = 0.5
        features = ['adj_close']

    



    if st.sidebar.button('Train Model'):
        file_name = load_data_from_s3(stock_name)
        st.markdown('## Training Model...')
        model, evaluations, df = run_model(
            stock_name=stock_name,
            ma_window=ma_window,
            ema_window=ema_window,
            sto_window=sto_window,
            features=features,
            test_size=test_size,
            future_days=future_days,
            rmse=rmse,
            mse=mse,
            mape=mape,
            display_at=display_at,
            alpha=alpha
        )

        if model is not None and evaluations is not None and df is not None:
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
        else:
            st.error('An error occurred during model training')

             # Display explanations on the main page
    st.markdown('## Explanations')
    st.markdown('**Evaluation Metrics**: Measures used to assess how well the model\'s predictions match the actual values.')
    st.markdown('- **RMSE (Root Mean Squared Error)**: A measure of the differences between the values predicted by the model and the actual values. Smaller values are better, with 0 being a perfect match.')
    st.markdown('- **MSE (Mean Squared Error)**: Similar to RMSE, but without taking the square root. This means larger errors are more heavily penalized.')
    st.markdown('- **MAPE (Mean Absolute Percentage Error)**: The average of the absolute percentage differences between the predicted and actual values. It gives an idea of the error rate in terms of the actual values.')
    
    if advanced_settings:
        st.markdown('**Alpha**: This is a value that helps determine how much weight the model gives to recent data points. A smaller alpha means the model considers older data more heavily, while a larger alpha means the model focuses more on recent data.')
        st.markdown('**Window Size**: This is the number of consecutive data points used to calculate the feature. For example, if the window size is 5, the feature for the current day will be calculated using the data from the current day and the 4 previous days:')
        st.markdown('- **Moving Average**: This is the average stock price over the specified window of days. It helps to smooth out price fluctuations and highlight the overall trend.')
        st.markdown('- **Exponential Moving Average Window Size**: Similar to the moving average, but it gives more weight to recent prices. This makes it react more quickly to price changes..')
        st.markdown('- **Stochastic Oscillator Window Size**: This is a momentum indicator that compares a particular closing price of a stock to a range of its prices over a certain period of time. The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result..')

if __name__ == '__main__':
    main()