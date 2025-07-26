import boto3
from time import sleep
from prefect_aws import S3Bucket, AwsCredentials


def create_aws_creds_block(profile_name, creds):
    aws_creds_block_obj = AwsCredentials(
        profile_name=profile_name,
        aws_access_key_id=creds.access_key,
        aws_secret_access_key=creds.secret_key,
        aws_session_token=creds.token
    )
    aws_creds_block_obj.save("mlops-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("mlops-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="mlflow-artifacts-remote-hiscox", 
        credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="mlops-s3-bucket", overwrite=True)


if __name__ == "__main__":

    profile_name = "mlops-user" 
    session = boto3.Session(profile_name=profile_name)
    creds = session.get_credentials().get_frozen_credentials()

    create_aws_creds_block(profile_name, creds)
    sleep(5)
    create_s3_bucket_block()
