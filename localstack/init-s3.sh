# localstack/init-s3.sh
#!/bin/bash

# Create the bucket
awslocal s3 mb s3://local-bucket

# Set bucket policy to public (for development only)
awslocal s3api put-bucket-acl --bucket local-bucket --acl public-read-write

echo "LocalStack S3 initialized with bucket: local-bucket"