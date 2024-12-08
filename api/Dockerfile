# api/Dockerfile
FROM golang:1.20-alpine

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN go build -o main .

# Create necessary directories
RUN mkdir -p models data

EXPOSE 8080

CMD ["./main"]