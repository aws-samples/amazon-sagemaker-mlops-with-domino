import os
import boto3
import botocore
import json
import csv
import uuid
from datetime import date

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
S3_BUCKET = os.environ['S3_BUCKET']
S3_KEY = os.environ['S3_KEY']

sagemaker = boto3.client('runtime.sagemaker')
s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')


def lambda_handler(event, context):
    # Retrieves the bucket and key of the inference data from the S3 Event
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = record['s3']['object']['key']

    csvfile = s3.get_object(Bucket=bucket, Key=key)
    lines = csvfile['Body'].read().decode('utf-8').split()

    # Skips the header, extracts the first row, and then stringify the list
    reader = csv.reader(lines)
    next(reader)
    row = ",".join(next(reader))

    # call SageMaker endpoint to get the prediction result, and then add id and date
    response = sagemaker.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                         ContentType='text/csv',
                                         Body=row)
    row_id = uuid.uuid4()
    result = str(response['Body'].read().decode("utf-8")).replace("\n", "")
    today_date = date.today().strftime("%m/%d/%y")
    result = row + ',' + str(row_id) + ',' + result + ',' + today_date

    # If prediction csv file exists, retrieve and append to it, otherwise create a new csv file
    try:
        prediction_csv = s3_resource.Bucket(S3_BUCKET).download_file(S3_KEY, '/tmp/prediction.csv')
    except botocore.exceptions.ClientError as e:
        if e.response['ResponseMetadata']['HTTPStatusCode'] == 404:
            option = "w+"
    else:
        option = "a+"

    # write the prediction result into the csv file
    with open("/tmp/prediction.csv", option) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(result.split(','))
        csv_file.close()

    # upload the csv with the new prediction to S3
    response_s3_put = s3.upload_file('/tmp/prediction.csv', S3_BUCKET, S3_KEY)

    return result
