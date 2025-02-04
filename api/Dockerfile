FROM golang:1.21-alpine AS builder

WORKDIR /app

# Print Go version for debugging
RUN go version

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

RUN export AWS_ACCESS_KEY_ID=fakeAccessKey
RUN export AWS_SECRET_ACCESS_KEY=fakeSecretKey
RUN export AWS_SESSION_TOKEN=fakeSessionToken
RUN export AWS_REGION=us-east-1
RUN export AWS_ENDPOINT_URL=http://localstack:4566

# Copy source code
COPY . .

# Build the application
RUN go build -o main .

# Create necessary directories
RUN mkdir -p models data

EXPOSE 8080

CMD ["./main"]