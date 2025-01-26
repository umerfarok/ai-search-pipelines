#!/bin/bash
# /docker-entrypoint-initaws.d/init-s3.sh

# Wait for LocalStack to be ready
until curl -s http://localhost:4566/_localstack/health | grep -q "\"s3\": \"running\""; do
    echo "Waiting for LocalStack S3 to be ready..."
    sleep 1
done

# Create bucket
awslocal s3 mb s3://product-search

# Enable versioning (optional)
awslocal s3api put-bucket-versioning \
    --bucket product-search \
    --versioning-configuration Status=Enabled

# Configure CORS (if needed)
awslocal s3api put-bucket-cors \
    --bucket product-search \
    --cors-configuration '{
        "CORSRules": [
            {
                "AllowedHeaders": ["*"],
                "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
                "AllowedOrigins": ["*"],
                "ExposeHeaders": []
            }
        ]
    }'

echo "S3 bucket 'product-search' initialized successfully"