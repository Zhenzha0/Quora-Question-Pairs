import os
import boto3
from dotenv import load_dotenv

load_dotenv()

endpoint_url = os.getenv("R2_ENDPOINT_URL")
access_key_id = os.getenv("R2_ACCESS_KEY_ID")
secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
bucket_name = os.getenv("R2_BUCKET")

if not all([endpoint_url, access_key_id, secret_access_key, bucket_name]):
    raise ValueError("Missing one or more required environment variables in .env")

s3 = boto3.client(
    service_name="s3",
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name="auto",
)

# Upload a file
s3.upload_file("myfile.txt", bucket_name, "myfile.txt")
print("Uploaded myfile.txt")


# List objects
response = s3.list_objects_v2(Bucket=bucket_name)
for obj in response.get("Contents", []):
    print(f"Object: {obj['Key']}")