package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

type LLMConfig struct {
	ID             primitive.ObjectID `bson:"_id" json:"id"`
	BaseModel      string             `bson:"base_model" json:"base_model"`
	TrainingConfig struct {
		BatchSize    int     `bson:"batch_size" json:"batch_size"`
		Epochs       int     `bson:"epochs" json:"epochs"`
		LearningRate float64 `bson:"learning_rate" json:"learning_rate"`
	} `bson:"training_config" json:"training_config"`
	DataSource struct {
		Location string `bson:"location" json:"location"`
		Type     string `bson:"type" json:"type"`
	} `bson:"data_source" json:"data_source"`
	Status        string    `bson:"status" json:"status"`
	CreatedAt     time.Time `bson:"created_at" json:"created_at"`
	UpdatedAt     time.Time `bson:"updated_at" json:"updated_at"`
	Version       string    `bson:"version" json:"version"`
	ModelPath     string    `bson:"model_path" json:"model_path"`
	TrainingStats struct {
		Progress  float64   `bson:"progress" json:"progress"`
		StartTime time.Time `bson:"start_time" json:"start_time"`
		EndTime   time.Time `bson:"end_time,omitempty" json:"end_time,omitempty"`
		Error     string    `bson:"error,omitempty" json:"error,omitempty"`
	} `bson:"training_stats" json:"training_stats"`
	IDHex        string    `bson:"-" json:"id_hex,omitempty"`
}

type LLMHandler struct {
	db          *mongo.Database
	redisClient *redis.Client
	s3Client    *s3.Client
}

func NewLLMHandler(db *mongo.Database, redisClient *redis.Client, s3Client *s3.Client) *LLMHandler {
	return &LLMHandler{
		db:          db,
		redisClient: redisClient,
		s3Client:    s3Client,
	}
}

func (h *LLMHandler) CreateTrainingConfig(c *gin.Context) {
	// Parse multipart form
	if err := c.Request.ParseMultipartForm(32 << 20); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("failed to parse form: %v", err)})
		return
	}

	// Get and validate config data
	configData := c.Request.FormValue("config")
	if configData == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "config data is required"})
		return
	}

	var config LLMConfig
	if err := json.Unmarshal([]byte(configData), &config); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid config format: %v", err)})
		return
	}

	// Get and validate file
	file, header, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("file is required: %v", err)})
		return
	}
	defer file.Close()

	// Initialize configuration
	config.ID = primitive.NewObjectID()
	config.CreatedAt = time.Now().UTC()
	config.UpdatedAt = config.CreatedAt
	config.Status = "pending"
	config.Version = time.Now().Format("20060102150405")
	config.ModelPath = fmt.Sprintf("models/%s", config.ID.Hex())

	// Upload file to S3
	s3Key := path.Join("training-data", config.ID.Hex(), header.Filename)
	if err := h.uploadToS3(file, s3Key); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to upload file: %v", err)})
		return
	}

	// Update config with file location
	config.DataSource.Location = s3Key
	config.DataSource.Type = path.Ext(header.Filename)[1:]

	// Save config to MongoDB
	if err := h.saveConfig(&config); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to save config: %v", err)})
		return
	}

	// Queue training job
	if err := h.queueTrainingJob(&config); err != nil {
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

func (h *LLMHandler) uploadToS3(file io.Reader, key string) error {
	ctx := context.Background()

	// Create uploader
	uploader := manager.NewUploader(h.s3Client)

	// Upload file
	_, err := uploader.Upload(ctx, &s3.PutObjectInput{
		Bucket: aws.String(os.Getenv("S3_BUCKET")),
		Key:    aws.String(key),
		Body:   file,
	})

	return err
}

func (h *LLMHandler) saveConfig(config *LLMConfig) error {
	ctx := context.Background()
	_, err := h.db.Collection("llm_configs").InsertOne(ctx, config)
	return err
}

func (h *LLMHandler) queueTrainingJob(config *LLMConfig) error {
	ctx := context.Background()

	// Convert config to JSON
	jobData, err := json.Marshal(config)
	if err != nil {
		return err
	}

	// Push to Redis queue
	err = h.redisClient.RPush(ctx, "llm-training-queue", jobData).Err()
	return err
}

func (h *LLMHandler) GetTrainingStatus(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "id is required"})
		return
	}

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid id format"})
		return
	}

	var config LLMConfig
	err = h.db.Collection("llm_configs").FindOne(
		context.Background(),
		bson.M{"_id": objID},
	).Decode(&config)

	if err == mongo.ErrNoDocuments {
		c.JSON(http.StatusNotFound, gin.H{"error": "config not found"})
		return
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get config"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data":   config,
	})
}
func (h *LLMHandler) ListModels(c *gin.Context) {
	var models []LLMConfig

	// Create pipeline for aggregation
	pipeline := mongo.Pipeline{
		// Match only completed models or those still processing
		{{"$match", bson.M{
			"status": bson.M{
				"$in": []string{"completed", "processing", "failed"},
			},
		}}},
		// Sort by creation date, newest first
		{{"$sort", bson.M{"created_at": -1}}},
		// Add training stats
		{{"$lookup", bson.M{
			"from":         "training_stats",
			"localField":   "_id",
			"foreignField": "model_id",
			"as":           "training_stats",
		}}},
		// Unwind training stats (converts array to object)
		{{"$unwind", bson.M{
			"path":                       "$training_stats",
			"preserveNullAndEmptyArrays": true,
		}}},
	}

	cursor, err := h.db.Collection("llm_configs").Aggregate(context.Background(), pipeline)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("failed to fetch models: %v", err),
		})
		return
	}
	defer cursor.Close(context.Background())

	if err = cursor.All(context.Background(), &models); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("failed to decode models: %v", err),
		})
		return
	}

	// Add model to router group
	for i := range models {
		// Add a new field to hold the hex string representation of the ObjectID
		models[i].IDHex = models[i].ID.Hex()

		// Check Redis for latest status
		statusKey := fmt.Sprintf("llm_model_status:%s", models[i].ID)
		status, err := h.redisClient.Get(context.Background(), statusKey).Result()
		if err == nil {
			// If found in Redis, update status
			var statusData map[string]interface{}
			if err := json.Unmarshal([]byte(status), &statusData); err == nil {
				models[i].Status = statusData["status"].(string)
				if stats, ok := statusData["training_stats"].(map[string]interface{}); ok {
					models[i].TrainingStats.Progress = stats["progress"].(float64)
				}
			}
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data":   models,
	})
}

// Chat session handlers
type ChatSession struct {
	ID        primitive.ObjectID `bson:"_id" json:"id"`
	ModelID   primitive.ObjectID `bson:"model_id" json:"model_id"`
	UserID    string             `bson:"user_id" json:"user_id"`
	CreatedAt time.Time          `bson:"created_at" json:"created_at"`
	UpdatedAt time.Time          `bson:"updated_at" json:"updated_at"`
	Messages  []ChatMessage      `bson:"messages" json:"messages"`
	Status    string             `bson:"status" json:"status"`
}

type ChatMessage struct {
	Role      string    `bson:"role" json:"role"`
	Content   string    `bson:"content" json:"content"`
	Timestamp time.Time `bson:"timestamp" json:"timestamp"`
}

func (h *LLMHandler) CreateChatSession(c *gin.Context) {
	var request struct {
		ModelID string `json:"model_id"`
		UserID  string `json:"user_id"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
		return
	}

	modelID, err := primitive.ObjectIDFromHex(request.ModelID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid model id"})
		return
	}

	session := ChatSession{
		ID:        primitive.NewObjectID(),
		ModelID:   modelID,
		UserID:    request.UserID,
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
		Status:    "active",
		Messages:  []ChatMessage{},
	}

	_, err = h.db.Collection("chat_sessions").InsertOne(context.Background(), session)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create session"})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"status": "success",
		"data": gin.H{
			"session_id": session.ID.Hex(),
		},
	})
}

func (h *LLMHandler) SendMessage(c *gin.Context) {
	var request struct {
		SessionID string `json:"session_id"`
		Message   string `json:"message"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
		return
	}

	sessionID, err := primitive.ObjectIDFromHex(request.SessionID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid session id"})
		return
	}

	// Get session
	var session ChatSession
	err = h.db.Collection("chat_sessions").FindOne(
		context.Background(),
		bson.M{"_id": sessionID},
	).Decode(&session)

	if err == mongo.ErrNoDocuments {
		c.JSON(http.StatusNotFound, gin.H{"error": "session not found"})
		return
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get session"})
		return
	}

	// Add user message
	userMsg := ChatMessage{
		Role:      "user",
		Content:   request.Message,
		Timestamp: time.Now().UTC(),
	}

	// Call chat service
	chatResponse, err := h.sendToChatService(session.ModelID.Hex(), request.Message)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get response"})
		return
	}

	// Add assistant message
	assistantMsg := ChatMessage{
		Role:      "assistant",
		Content:   chatResponse,
		Timestamp: time.Now().UTC(),
	}

	// Update session with new messages
	update := bson.M{
		"$push": bson.M{
			"messages": bson.M{
				"$each": []ChatMessage{userMsg, assistantMsg},
			},
		},
		"$set": bson.M{
			"updated_at": time.Now().UTC(),
		},
	}

	_, err = h.db.Collection("chat_sessions").UpdateOne(
		context.Background(),
		bson.M{"_id": sessionID},
		update,
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update session"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data": gin.H{
			"response": chatResponse,
		},
	})
}

func (h *LLMHandler) sendToChatService(modelID string, message string) (string, error) {
	// Make request to chat service
	chatServiceURL := fmt.Sprintf("%s/chat/message", os.Getenv("CHAT_SERVICE_URL"))

	requestBody := map[string]string{
		"model_id": modelID,
		"message":  message,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return "", err
	}

	resp, err := http.Post(
		chatServiceURL,
		"application/json",
		bytes.NewBuffer(jsonBody),
	)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("chat service returned status %d", resp.StatusCode)
	}

	var response struct {
		Data struct {
			Response string `json:"response"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", err
	}

	return response.Data.Response, nil
}

func (h *LLMHandler) GetSession(c *gin.Context) {
	sessionID, err := primitive.ObjectIDFromHex(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid session id"})
		return
	}

	var session ChatSession
	err = h.db.Collection("chat_sessions").FindOne(
		context.Background(),
		bson.M{"_id": sessionID},
	).Decode(&session)

	if err == mongo.ErrNoDocuments {
		c.JSON(http.StatusNotFound, gin.H{"error": "session not found"})
		return
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get session"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data":   session,
	})
}
