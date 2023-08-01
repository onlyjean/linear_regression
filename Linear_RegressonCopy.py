import streamlit as st
import pandas as pd
import boto3
import os
import json

# Load AWS credentials
with open('/Users/cedrix/Documents/aws.json', 'r') as f:
    credentials = json.load(f)

# Set environment variables
os.environ['AWS_ACCESS_KEY_ID'] = credentials ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = credentials ['AWS_SECRET_ACCESS_KEY']

# AWS S3 bucket
bucket = 'raw-stock-price'

# Function to load data from S3
def load_data_from_s3(file_name):
    s3 = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])
    obj = s3.get_object(Bucket=bucket, Key=file_name)
    df = pd.read_csv(obj['Body'])
    return df

def main():
    st.title('S3 File Read Test')

    # Specify the file name
    file_name = 'yhoofinance-daily-historical-data/TSLA_daily_data.csv'

    try:
        # Load data from S3
        df = load_data_from_s3(file_name)
        st.write(df)
    except Exception as e:
        st.write(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
