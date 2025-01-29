#!/bin/bash
# /docker-entrypoint-initaws.d/init-s3.sh

# Wait for LocalStack to be ready
until curl -s http://localhost:4566/_localstack/health | grep '"s3": "running"' > /dev/null; do
    echo "Waiting for LocalStack S3..."
    sleep 1
done

# Create the bucket
awslocal s3 mb s3://local-bucket

# Set bucket policy for public access
awslocal s3api put-bucket-policy --bucket local-bucket --policy '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadWrite",
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:*"],
            "Resource": ["arn:aws:s3:::local-bucket", "arn:aws:s3:::local-bucket/*"]
        }
    ]
}'

echo "S3 bucket 'local-bucket' created and configured"