import os
import boto3
import json
import urllib3

# grab environment variables
S3_BUCKET = os.environ['S3_BUCKET']
PRED_FILE_KEY = os.environ['PRED_FILE_KEY']
DMM_URL = os.environ['DMM_URL']
DMM_API_TOKEN = os.environ['DMM_API_TOKEN']
DMM_HEADER = {'Authorization': DMM_API_TOKEN, 'Content-Type': 'application/json'}
MODEL_ID = os.environ['MODEL_ID']

s3 = boto3.client('s3')


def lambda_handler(event, context):
    http = urllib3.PoolManager()

    # PUT prediction data to our model
    full_prediction_data_url = 'https://' + S3_BUCKET + '.s3.amazonaws.com/' + PRED_FILE_KEY
    dmm_prediction_api = DMM_URL + MODEL_ID + '/add_predictions'
    prediction_data = {"dataLocation": full_prediction_data_url, }
    encoded_data = json.dumps(prediction_data).encode('utf-8')
    response_pred = http.request(
        'PUT',
        dmm_prediction_api,
        body=encoded_data,
        headers=DMM_HEADER
    )

    print(response_pred.status)