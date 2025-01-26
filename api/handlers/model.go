package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	typecfg "github.com/umerfarok/product-search/config"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

const (
	trainingQueueKey  = "training_queue"
	modelStatusPrefix = "model_status:"
)

type ConfigService struct {
	db          *mongo.Database
	s3Client    *s3.Client
	s3Bucket    string
	redisClient *redis.Client
}

func NewConfigService(db *mongo.Database) (*ConfigService, error) {
	log.Println("Initializing ConfigService")

	// Initialize Redis client with more robust configuration
	redisClient := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%s",
			os.Getenv("REDIS_HOST"),
			os.Getenv("REDIS_PORT")),
		Password: os.Getenv("REDIS_PASSWORD"), // Add password if needed
		DB:       0,
		OnConnect: func(ctx context.Context, cn *redis.Conn) error {
			log.Println("Connected to Redis")
			return nil
		},
	})

	// Test Redis connection
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %v", err)
	}

	// Initialize S3 client with custom resolver for LocalStack support
	customResolver := aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...interface{}) (aws.Endpoint, error) {
		if endpoint := os.Getenv("AWS_ENDPOINT_URL"); endpoint != "" {
			return aws.Endpoint{
				PartitionID:       "aws",
				URL:               endpoint,
				SigningRegion:     os.Getenv("AWS_REGION"),
				HostnameImmutable: true,
			}, nil
		}
		return aws.Endpoint{}, &aws.EndpointNotFoundError{}
	})

	cfg, err := config.LoadDefaultConfig(context.TODO(),
		config.WithRegion(os.Getenv("AWS_REGION")),
		config.WithEndpointResolverWithOptions(customResolver),
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
			os.Getenv("AWS_ACCESS_KEY"),
			os.Getenv("AWS_SECRET_KEY"),
			"",
		)),
	)
	if err != nil {
		return nil, fmt.Errorf("unable to load SDK config: %v", err)
	}

	s3Client := s3.NewFromConfig(cfg, func(o *s3.Options) {
		o.UsePathStyle = true
	})

	return &ConfigService{
		db:          db,
		s3Client:    s3Client,
		s3Bucket:    os.Getenv("S3_BUCKET"),
		redisClient: redisClient,
	}, nil
}

func (s *ConfigService) CreateConfigWithData(c *gin.Context) {
	// Parse multipart form with larger size limit
	if err := c.Request.ParseMultipartForm(50 << 20); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("failed to parse form: %v", err)})
		return
	}

	// Get and validate config data
	configData := c.Request.FormValue("config")
	if configData == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "config data is required"})
		return
	}

	var cfg typecfg.ProductConfig
	if err := json.Unmarshal([]byte(configData), &cfg); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid config format: %v", err)})
		return
	}

	// Validate required fields
	if err := validateConfig(&cfg); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Get and validate file
	file, header, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("file is required: %v", err)})
		return
	}
	defer file.Close()

	if !isValidCSVFile(header) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid file type, only CSV files are allowed"})
		return
	}

	// Generate IDs and prepare metadata
	configID := primitive.NewObjectID()
	versionID := primitive.NewObjectID()
	timeNow := time.Now().UTC().Format(time.RFC3339)

	// Prepare S3 paths
	s3Key := path.Join("data", configID.Hex(), header.Filename)
	modelPath := fmt.Sprintf("models/%s", versionID.Hex())

	// Upload file to S3
	if err := s.uploadFileToS3(file, s3Key); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to upload file: %v", err)})
		return
	}

	// Update config with metadata
	cfg.ID = configID
	cfg.CreatedAt = timeNow
	cfg.UpdatedAt = timeNow
	cfg.Status = string(typecfg.ModelStatusPending)
	cfg.DataSource.Location = s3Key

	// Create model version
	version := typecfg.ModelVersion{
		ID:        versionID,
		ConfigID:  configID,
		Version:   time.Now().Format("20060102150405"),
		Status:    string(typecfg.ModelStatusPending),
		S3Path:    modelPath,
		CreatedAt: timeNow,
		UpdatedAt: timeNow,
		Config:    cfg, // Store complete config for reference
	}

	// Save config first
	if _, err := s.db.Collection("configs").InsertOne(context.Background(), cfg); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to save config: %v", err)})
		return
	}

	// Save version
	if _, err := s.db.Collection("model_versions").InsertOne(context.Background(), version); err != nil {
		// If version save fails, attempt to rollback config
		if _, deleteErr := s.db.Collection("configs").DeleteOne(context.Background(), bson.M{"_id": configID}); deleteErr != nil {
			log.Printf("Warning: Failed to rollback config after version save error: %v", deleteErr)
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to save model version: %v", err)})
		return
	}

	// Prepare training job
	job := typecfg.QueuedJob{
		VersionID: versionID.Hex(),
		ConfigID:  configID.Hex(),
		Config: map[string]interface{}{
			"schema_mapping": cfg.SchemaMapping,
			"data_source": map[string]interface{}{
				"type":     cfg.DataSource.Type,
				"location": fmt.Sprintf("s3://%s/%s", s.s3Bucket, s3Key),
			},
			"mode":            cfg.Mode,
			"training_config": cfg.TrainingConfig,
		},
		S3Path: modelPath,
	}

	// Save initial status in Redis
	statusKey := fmt.Sprintf("%s%s", modelStatusPrefix, versionID.Hex())
	initialStatus := map[string]interface{}{
		"status":    typecfg.ModelStatusPending,
		"timestamp": timeNow,
	}

	statusBytes, _ := json.Marshal(initialStatus)
	if err := s.redisClient.Set(context.Background(), statusKey, statusBytes, 24*time.Hour).Err(); err != nil {
		log.Printf("Warning: Failed to set initial status in Redis: %v", err)
	}

	// Queue training job
	jobBytes, _ := json.Marshal(job)
	if err := s.redisClient.LPush(context.Background(), trainingQueueKey, jobBytes).Err(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to queue job: %v", err)})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"status": "success",
		"data": gin.H{
			"config_id":  configID.Hex(),
			"version_id": versionID.Hex(),
			"file_path":  cfg.DataSource.Location,
			"model_path": modelPath,
		},
	})
}
func validateConfig(cfg *typecfg.ProductConfig) error {
	if cfg.Name == "" {
		return fmt.Errorf("name is required")
	}
	if cfg.DataSource.Type == "" {
		return fmt.Errorf("data_source.type is required")
	}
	if cfg.Mode == "" {
		cfg.Mode = "replace"
	}
	if cfg.TrainingConfig.BatchSize == 0 {
		cfg.TrainingConfig.BatchSize = 128
	}
	return nil
}

func (s *ConfigService) uploadFileToS3(file multipart.File, key string) error {
	content, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("failed to read file: %v", err)
	}

	_, err = file.Seek(0, 0)
	if err != nil {
		return fmt.Errorf("failed to reset file pointer: %v", err)
	}

	_, err = s.s3Client.PutObject(context.TODO(), &s3.PutObjectInput{
		Bucket:      aws.String(s.s3Bucket),
		Key:         aws.String(key),
		Body:        bytes.NewReader(content),
		ContentType: aws.String("text/csv"),
	})

	if err != nil {
		return fmt.Errorf("failed to upload to S3: %v", err)
	}

	return nil
}

func isValidCSVFile(header *multipart.FileHeader) bool {
	ext := path.Ext(header.Filename)
	return ext == ".csv"
}

func (s *ConfigService) GetTrainingStatus(c *gin.Context) {
	versionID := c.Param("id")
	if versionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "version_id is required"})
		return
	}

	// Check Redis status first
	statusKey := fmt.Sprintf("%s%s", modelStatusPrefix, versionID)
	statusData, err := s.redisClient.Get(context.Background(), statusKey).Bytes()
	if err == nil {
		var status map[string]interface{}
		if err := json.Unmarshal(statusData, &status); err == nil {
			c.JSON(http.StatusOK, status)
			return
		}
	}

	// Fall back to database status
	id, err := primitive.ObjectIDFromHex(versionID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid version_id format"})
		return
	}

	var version typecfg.ModelVersion
	err = s.db.Collection("model_versions").FindOne(
		context.Background(),
		bson.M{"_id": id},
	).Decode(&version)

	if err != nil {
		if err == mongo.ErrNoDocuments {
			c.JSON(http.StatusNotFound, gin.H{"error": "model version not found"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":     version.Status,
		"error":      version.Error,
		"updated_at": version.UpdatedAt,
	})
}

func (s *ConfigService) UpdateModelVersionStatus(c *gin.Context) {
	versionID := c.Param("id")
	if versionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "version_id is required"})
		return
	}

	var updateReq struct {
		Status    string `json:"status" binding:"required"`
		Error     string `json:"error,omitempty"`
		UpdatedAt string `json:"updated_at"`
	}

	if err := c.ShouldBindJSON(&updateReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Update Redis status
	statusKey := fmt.Sprintf("%s%s", modelStatusPrefix, versionID)
	status := map[string]interface{}{
		"status":     updateReq.Status,
		"error":      updateReq.Error,
		"updated_at": updateReq.UpdatedAt,
	}

	statusBytes, _ := json.Marshal(status)
	if err := s.redisClient.Set(context.Background(), statusKey, statusBytes, 24*time.Hour).Err(); err != nil {
		log.Printf("Warning: Failed to update Redis status: %v", err)
	}

	// Update MongoDB
	objID, err := primitive.ObjectIDFromHex(versionID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid version id"})
		return
	}

	update := bson.M{
		"$set": bson.M{
			"status":     updateReq.Status,
			"error":      updateReq.Error,
			"updated_at": updateReq.UpdatedAt,
		},
	}

	result, err := s.db.Collection("model_versions").UpdateOne(
		context.Background(),
		bson.M{"_id": objID},
		update,
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update version"})
		return
	}

	if result.MatchedCount == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "version not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "updated"})
}

func (s *ConfigService) GetQueuedJobs(c *gin.Context) {
	// Get all jobs from Redis queue without removing them
	queueLen, err := s.redisClient.LLen(context.Background(), trainingQueueKey).Result()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to get queue length: %v", err)})
		return
	}

	if queueLen == 0 {
		c.JSON(http.StatusOK, gin.H{
			"count": 0,
			"jobs":  []interface{}{},
		})
		return
	}

	// Get all jobs using LRANGE
	jobsData, err := s.redisClient.LRange(context.Background(), trainingQueueKey, 0, queueLen-1).Result()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to get jobs: %v", err)})
		return
	}

	// Parse each job
	var jobs []typecfg.QueuedJob
	for _, jobData := range jobsData {
		var job typecfg.QueuedJob
		if err := json.Unmarshal([]byte(jobData), &job); err != nil {
			log.Printf("Error parsing job data: %v", err)
			continue
		}
		jobs = append(jobs, job)
	}

	// Return jobs with count
	c.JSON(http.StatusOK, gin.H{
		"count": len(jobs),
		"jobs":  jobs,
	})
}

// Helper method to ensure Redis and MongoDB are in sync
func (s *ConfigService) syncStatus(versionID string) error {
	ctx := context.Background()

	// Get status from MongoDB
	objID, err := primitive.ObjectIDFromHex(versionID)
	if err != nil {
		return err
	}

	var version typecfg.ModelVersion
	err = s.db.Collection("model_versions").FindOne(ctx, bson.M{"_id": objID}).Decode(&version)
	if err != nil {
		return err
	}

	// Update Redis with MongoDB status
	status := map[string]interface{}{
		"status":     version.Status,
		"error":      version.Error,
		"updated_at": version.UpdatedAt,
	}

	statusBytes, _ := json.Marshal(status)
	statusKey := fmt.Sprintf("%s%s", modelStatusPrefix, versionID)
	return s.redisClient.Set(ctx, statusKey, statusBytes, 24*time.Hour).Err()
}
