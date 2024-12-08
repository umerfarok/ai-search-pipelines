// api/handlers/model.go

package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/umerfarok/product-search/config"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type TrainingService struct {
	db           *mongo.Database
	trainingHost string
}

func NewTrainingService(db *mongo.Database) *TrainingService {
	trainingHost := os.Getenv("TRAINING_SERVICE_HOST")
	if trainingHost == "" {
		trainingHost = "http://localhost:5001"
	}

	return &TrainingService{
		db:           db,
		trainingHost: trainingHost,
	}
}

func (s *TrainingService) TriggerTraining(c *gin.Context) {
	var req struct {
		ConfigID string `json:"config_id"`
	}
	if err := c.BindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "config_id is required"})
		return
	}

	configID, err := primitive.ObjectIDFromHex(req.ConfigID)
	if err != nil {
		c.JSON(400, gin.H{"error": "invalid config_id format"})
		return
	}

	var cfg config.ProductConfig
	if err := s.db.Collection("configs").FindOne(context.Background(), bson.M{"_id": configID}).Decode(&cfg); err != nil {
		if err == mongo.ErrNoDocuments {
			c.JSON(404, gin.H{"error": "configuration not found"})
			return
		}
		c.JSON(500, gin.H{"error": "failed to fetch configuration"})
		return
	}

	// Create model version
	version := config.ModelVersion{
		ID:        primitive.NewObjectID(),
		ConfigID:  configID,
		Version:   time.Now().Format("20060102150405"),
		Status:    "training",
		CreatedAt: time.Now().UTC(),
	}

	if _, err := s.db.Collection("model_versions").InsertOne(context.Background(), version); err != nil {
		c.JSON(500, gin.H{"error": "failed to create model version"})
		return
	}

	// Create output directory
	outputDir := filepath.Join("models", version.Version)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		s.updateModelStatus(version.ID, "failed", fmt.Sprintf("failed to create output directory: %v", err))
		c.JSON(500, gin.H{"error": "failed to create output directory"})
		return
	}

	// Start training asynchronously
	go s.runTraining(cfg, version, outputDir)

	c.JSON(202, version)
}

func (s *TrainingService) runTraining(cfg config.ProductConfig, version config.ModelVersion, outputDir string) {
	requestBody := map[string]interface{}{
		"training_id": version.ID.Hex(),
		"config":      cfg,
		"output_dir":  outputDir,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		s.updateModelStatus(version.ID, "failed", fmt.Sprintf("failed to marshal request: %v", err))
		return
	}

	resp, err := http.Post(fmt.Sprintf("%s/train", s.trainingHost), "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		s.updateModelStatus(version.ID, "failed", fmt.Sprintf("failed to start training: %v", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		s.updateModelStatus(version.ID, "failed", fmt.Sprintf("training service error: %s", string(body)))
		return
	}

	// Monitor training status
	s.monitorTraining(version.ID)
}

func (s *TrainingService) monitorTraining(versionID primitive.ObjectID) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	maxAttempts := 360 // 1 hour maximum (10 seconds * 360)
	attempts := 0

	for {
		select {
		case <-ticker.C:
			attempts++
			if attempts > maxAttempts {
				s.updateModelStatus(versionID, "failed", "training timeout exceeded")
				return
			}

			status, err := s.checkTrainingStatus(versionID)
			if err != nil {
				s.updateModelStatus(versionID, "failed", fmt.Sprintf("failed to check training status: %v", err))
				return
			}

			switch status["status"] {
			case "completed":
				s.updateModelStatus(versionID, "completed", "")
				return
			case "failed":
				s.updateModelStatus(versionID, "failed", status["error"].(string))
				return
			}
		}
	}
}

func (s *TrainingService) checkTrainingStatus(versionID primitive.ObjectID) (map[string]interface{}, error) {
	resp, err := http.Get(fmt.Sprintf("%s/training-status/%s", s.trainingHost, versionID.Hex()))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var status map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, err
	}

	return status, nil
}

func (s *TrainingService) updateModelStatus(versionID primitive.ObjectID, status string, errorMsg string) {
	update := bson.M{
		"$set": bson.M{
			"status": status,
			"error":  errorMsg,
		},
	}

	s.db.Collection("model_versions").UpdateOne(
		context.Background(),
		bson.M{"_id": versionID},
		update,
	)
}

func GetModelVersions(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: -1}})

		filter := bson.M{}
		if configID := c.Query("config_id"); configID != "" {
			objectID, err := primitive.ObjectIDFromHex(configID)
			if err != nil {
				c.JSON(400, gin.H{"error": "invalid config_id format"})
				return
			}
			filter["config_id"] = objectID
		}

		cursor, err := db.Collection("model_versions").Find(context.Background(), filter, opts)
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to fetch model versions"})
			return
		}
		defer cursor.Close(context.Background())

		var versions []config.ModelVersion
		if err := cursor.All(context.Background(), &versions); err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}

		c.JSON(200, versions)
	}
}

func GetModelVersion(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		id, err := primitive.ObjectIDFromHex(c.Param("id"))
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid id format"})
			return
		}

		var version config.ModelVersion
		err = db.Collection("model_versions").FindOne(
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

		c.JSON(200, version)
	}
}
