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
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	cfg "github.com/umerfarok/product-search/config"
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

func (s *ConfigService) CreateConfig(c *gin.Context) {
	// Parse multipart form
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

	var config cfg.ModelConfig
	if err := json.Unmarshal([]byte(configData), &config); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid config format: %v", err)})
		return
	}

	// Validate required fields
	if err := validateConfig(&config); err != nil {
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

	// Initialize configuration metadata
	config.ID = primitive.NewObjectID()
	timeNow := time.Now().UTC().Format(time.RFC3339)
	config.CreatedAt = timeNow
	config.UpdatedAt = timeNow
	config.Status = string(cfg.ModelStatusPending)
	config.Version = time.Now().Format("20060102150405")

	// Set model path based on ID and version
	config.ModelPath = fmt.Sprintf("models/%s", config.ID.Hex())

	// Handle append mode
	if config.Mode == "append" && config.PreviousVersion != "" {
		if err := s.validatePreviousVersion(config.PreviousVersion); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}

	// Upload file to S3
	s3Key := path.Join("data", config.ID.Hex(), header.Filename)
	if err := s.uploadFileToS3(file, s3Key); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to upload file: %v", err)})
		return
	}

	// Update config with file location
	config.DataSource.Location = s3Key

	// Initialize training stats
	config.TrainingStats = &cfg.TrainingStats{
		StartTime: timeNow,
		Progress:  0,
	}

	// Save config to MongoDB
	if err := s.saveConfig(&config); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to save config: %v", err)})
		return
	}

	// Queue training job
	if err := s.queueTrainingJob(&config); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to queue training job: %v", err)})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"status": "success",
		"data": gin.H{
			"config_id":  config.ID.Hex(),
			"version":    config.Version,
			"model_path": config.ModelPath,
		},
	})
}

func (s *ConfigService) GetConfig(c *gin.Context) {
	id, err := primitive.ObjectIDFromHex(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid id format"})
		return
	}

	var config cfg.ModelConfig
	err = s.db.Collection("configs").FindOne(
		context.Background(),
		bson.M{"_id": id},
	).Decode(&config)

	if err != nil {
		if err == mongo.ErrNoDocuments {
			c.JSON(http.StatusNotFound, gin.H{"error": "configuration not found"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, config)
}

func (s *ConfigService) ListConfigs(c *gin.Context) {
	filter := bson.M{}

	// Add filters
	if status := c.Query("status"); status != "" {
		filter["status"] = status
	}

	if mode := c.Query("mode"); mode != "" {
		filter["mode"] = mode
	}

	// Add pagination
	limit := 10
	if limitStr := c.Query("limit"); limitStr != "" {
		if val, err := strconv.Atoi(limitStr); err == nil {
			limit = val
		}
	}

	skip := 0
	if page := c.Query("page"); page != "" {
		if val, err := strconv.Atoi(page); err == nil {
			skip = (val - 1) * limit
		}
	}

	opts := options.Find().
		SetSort(bson.D{{Key: "created_at", Value: -1}}).
		SetLimit(int64(limit)).
		SetSkip(int64(skip))

	cursor, err := s.db.Collection("configs").Find(context.Background(), filter, opts)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to fetch configurations: %v", err)})
		return
	}
	defer cursor.Close(context.Background())

	var configs []cfg.ModelConfig
	if err := cursor.All(context.Background(), &configs); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Get total count for pagination
	total, err := s.db.Collection("configs").CountDocuments(context.Background(), filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get total count"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"configs": configs,
		"total":   total,
		"page":    skip/limit + 1,
		"limit":   limit,
	})
}

func (s *ConfigService) UpdateConfigStatus(c *gin.Context) {
	configID := c.Param("id")
	if configID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "config_id is required"})
		return
	}

	var updateReq struct {
		Status    string  `json:"status" binding:"required"`
		Error     string  `json:"error,omitempty"`
		Progress  float64 `json:"progress,omitempty"`
		UpdatedAt string  `json:"updated_at"`
	}

	if err := c.ShouldBindJSON(&updateReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Update Redis status for quick access
	statusKey := fmt.Sprintf("model_status:%s", configID)
	status := map[string]interface{}{
		"status":     updateReq.Status,
		"error":      updateReq.Error,
		"progress":   updateReq.Progress,
		"updated_at": updateReq.UpdatedAt,
	}

	statusBytes, _ := json.Marshal(status)
	if err := s.redisClient.Set(context.Background(), statusKey, statusBytes, 24*time.Hour).Err(); err != nil {
		log.Printf("Warning: Failed to update Redis status: %v", err)
	}

	// Update MongoDB
	objID, err := primitive.ObjectIDFromHex(configID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid config id"})
		return
	}

	update := bson.M{
		"$set": bson.M{
			"status":                  updateReq.Status,
			"error":                   updateReq.Error,
			"updated_at":              updateReq.UpdatedAt,
			"training_stats.progress": updateReq.Progress,
		},
	}

	result, err := s.db.Collection("configs").UpdateOne(
		context.Background(),
		bson.M{"_id": objID},
		update,
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update config"})
		return
	}

	if result.MatchedCount == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "config not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "updated"})
}

func (s *ConfigService) validatePreviousVersion(previousVersion string) error {
	id, err := primitive.ObjectIDFromHex(previousVersion)
	if err != nil {
		return fmt.Errorf("invalid previous version id")
	}

	var prevConfig cfg.ModelConfig
	err = s.db.Collection("configs").FindOne(
		context.Background(),
		bson.M{
			"_id":    id,
			"status": cfg.ModelStatusCompleted,
		},
	).Decode(&prevConfig)

	if err != nil {
		if err == mongo.ErrNoDocuments {
			return fmt.Errorf("previous version not found or not completed")
		}
		return fmt.Errorf("error validating previous version: %v", err)
	}

	return nil
}

func (s *ConfigService) saveConfig(config *cfg.ModelConfig) error {
	_, err := s.db.Collection("configs").InsertOne(context.Background(), config)
	return err
}

func (s *ConfigService) queueTrainingJob(config *cfg.ModelConfig) error {
	job := cfg.QueuedJob{
		ConfigID:  config.ID.Hex(),
		Config:    *config,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	jobBytes, err := json.Marshal(job)
	if err != nil {
		return fmt.Errorf("failed to marshal job: %v", err)
	}

	return s.redisClient.LPush(context.Background(), "training_queue", jobBytes).Err()
}

func validateConfig(config *cfg.ModelConfig) error {
	if config.Name == "" {
		return fmt.Errorf("name is required")
	}

	if config.DataSource.Type == "" {
		return fmt.Errorf("data_source.type is required")
	}

	if config.Mode != "replace" && config.Mode != "append" {
		return fmt.Errorf("mode must be either 'replace' or 'append'")
	}

	if config.Mode == "append" && config.PreviousVersion == "" {
		return fmt.Errorf("previous_version is required for append mode")
	}

	// Updated field names
	if config.SchemaMapping.Namecolumn == "" {
		return fmt.Errorf("schema_mapping.namecolumn is required")
	}

	if config.SchemaMapping.Idcolumn == "" {
		return fmt.Errorf("schema_mapping.idcolumn is required")
	}

	// Set default values
	if config.TrainingConfig.BatchSize == 0 {
		config.TrainingConfig.BatchSize = 128
	}

	return nil
}

func (s *ConfigService) GetTrainingStatus(c *gin.Context) {
	configID := c.Param("id")
	if configID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "config_id is required"})
		return
	}

	// Check Redis status first for performance
	statusKey := fmt.Sprintf("model_status:%s", configID)
	statusData, err := s.redisClient.Get(context.Background(), statusKey).Bytes()
	if err == nil {
		var status map[string]interface{}
		if err := json.Unmarshal(statusData, &status); err == nil {
			c.JSON(http.StatusOK, status)
			return
		}
	}

	// Fall back to MongoDB
	id, err := primitive.ObjectIDFromHex(configID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid config_id format"})
		return
	}

	var config cfg.ModelConfig
	err = s.db.Collection("configs").FindOne(
		context.Background(),
		bson.M{"_id": id},
	).Decode(&config)

	if err != nil {
		if err == mongo.ErrNoDocuments {
			c.JSON(http.StatusNotFound, gin.H{"error": "config not found"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Return status response
	c.JSON(http.StatusOK, gin.H{
		"status":         config.Status,
		"error":          config.Error,
		"progress":       config.TrainingStats.Progress,
		"updated_at":     config.UpdatedAt,
		"training_stats": config.TrainingStats,
	})
}

func (s *ConfigService) GetQueuedJobs(c *gin.Context) {
	queueLen, err := s.redisClient.LLen(context.Background(), "training_queue").Result()
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

	jobsData, err := s.redisClient.LRange(context.Background(), "training_queue", 0, queueLen-1).Result()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to get jobs: %v", err)})
		return
	}

	var jobs []cfg.QueuedJob
	for _, jobData := range jobsData {
		var job cfg.QueuedJob
		if err := json.Unmarshal([]byte(jobData), &job); err != nil {
			log.Printf("Error parsing job data: %v", err)
			continue
		}
		jobs = append(jobs, job)
	}

	c.JSON(http.StatusOK, gin.H{
		"count": len(jobs),
		"jobs":  jobs,
	})
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
func (s *ConfigService) GetAvailableLLMModels(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"models": cfg.AvailableLLMModels,
	})
}
