// api/handlers/training.go

package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	typCgf "github.com/umerfarok/product-search/config"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

const (
	trainingQueue     = "training_queue"
	modelStatusPrefix = "model_status:"
	statusTTL         = 24 * time.Hour
)

type TrainingService struct {
	db    *mongo.Database
	redis *redis.Client
}

type TrainingRequest struct {
	ConfigID string `json:"config_id" binding:"required"`
}

type TrainingStatus struct {
	Status    string    `json:"status"`
	Progress  float64   `json:"progress,omitempty"`
	Error     string    `json:"error,omitempty"`
	StartTime time.Time `json:"start_time,omitempty"`
	EndTime   time.Time `json:"end_time,omitempty"`
}

func NewTrainingService(db *mongo.Database) *TrainingService {
	redisHost := getEnv("REDIS_HOST", "localhost")
	redisPort := getEnv("REDIS_PORT", "6379")

	rdb := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%s", redisHost, redisPort),
		DB:   0,
	})

	return &TrainingService{
		db:    db,
		redis: rdb,
	}
}

func (s *TrainingService) TriggerTraining(c *gin.Context) {
	var req TrainingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Validate config exists
	configID, err := primitive.ObjectIDFromHex(req.ConfigID)
	if err != nil {
		c.JSON(400, gin.H{"error": "invalid config_id"})
		return
	}

	var config typCgf.ProductConfig
	if err := s.db.Collection("configs").FindOne(context.Background(), bson.M{"_id": configID}).Decode(&config); err != nil {
		if err == mongo.ErrNoDocuments {
			c.JSON(404, gin.H{"error": "configuration not found"})
			return
		}
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// Create model version
	version := typCgf.ModelVersion{
		ID:        primitive.NewObjectID(),
		ConfigID:  configID,
		Version:   time.Now().Format("20060102150405"),
		Status:    "queued",
		CreatedAt: time.Now(),
	}

	if _, err := s.db.Collection("model_versions").InsertOne(context.Background(), version); err != nil {
		c.JSON(500, gin.H{"error": "failed to create model version"})
		return
	}

	// Prepare training job
	trainingJob := map[string]interface{}{
		"version_id":  version.ID.Hex(),
		"config_id":   req.ConfigID,
		"config":      config,
		"output_path": filepath.Join("./models", version.Version),
		"create_time": time.Now(),
	}

	// Add to Redis queue
	jobBytes, err := json.Marshal(trainingJob)
	if err != nil {
		c.JSON(500, gin.H{"error": "failed to serialize training job"})
		return
	}

	if err := s.redis.LPush(context.Background(), trainingQueue, jobBytes).Err(); err != nil {
		c.JSON(500, gin.H{"error": "failed to queue training job"})
		return
	}

	// Set initial status
	status := TrainingStatus{
		Status:    "queued",
		StartTime: time.Now(),
	}
	statusKey := fmt.Sprintf("%s%s", modelStatusPrefix, version.ID.Hex())
	statusBytes, _ := json.Marshal(status)
	s.redis.Set(context.Background(), statusKey, statusBytes, statusTTL)

	c.JSON(202, version)
}

func (s *TrainingService) GetStatus(c *gin.Context) {
	versionID := c.Param("id")
	if versionID == "" {
		c.JSON(400, gin.H{"error": "version_id is required"})
		return
	}

	// Check Redis status first
	statusKey := fmt.Sprintf("%s%s", modelStatusPrefix, versionID)
	statusData, err := s.redis.Get(context.Background(), statusKey).Bytes()
	if err == nil {
		var status TrainingStatus
		if err := json.Unmarshal(statusData, &status); err == nil {
			c.JSON(200, status)
			return
		}
	}

	// Fall back to database status
	id, err := primitive.ObjectIDFromHex(versionID)
	if err != nil {
		c.JSON(400, gin.H{"error": "invalid version_id format"})
		return
	}

	var version typCgf.ModelVersion
	err = s.db.Collection("model_versions").FindOne(
		context.Background(),
		bson.M{"_id": id},
	).Decode(&version)

	if err != nil {
		if err == mongo.ErrNoDocuments {
			c.JSON(404, gin.H{"error": "model version not found"})
			return
		}
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	status := TrainingStatus{
		Status: version.Status,
		Error:  version.Error,
	}

	c.JSON(200, status)
}

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

func UpdateModelVersionStatus(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		versionID, err := primitive.ObjectIDFromHex(c.Param("id"))
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid version id"})
			return
		}

		var updateReq struct {
			Status    string `json:"status" binding:"required"`
			Error     string `json:"error,omitempty"`
			UpdatedAt string `json:"updated_at"`
		}

		if err := c.ShouldBindJSON(&updateReq); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		update := bson.M{
			"$set": bson.M{
				"status":     updateReq.Status,
				"error":      updateReq.Error,
				"updated_at": updateReq.UpdatedAt,
			},
		}

		result, err := db.Collection("model_versions").UpdateOne(
			context.Background(),
			bson.M{"_id": versionID},
			update,
		)

		if err != nil {
			c.JSON(500, gin.H{"error": "failed to update version"})
			return
		}

		if result.MatchedCount == 0 {
			c.JSON(404, gin.H{"error": "version not found"})
			return
		}

		c.JSON(200, gin.H{"status": "updated"})
	}
}
