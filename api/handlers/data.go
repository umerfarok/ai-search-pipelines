// api/handlers/data.go
package handlers

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

type DataUploadRequest struct {
	FileName string `json:"file_name" binding:"required"`
	Content  string `json:"content" binding:"required"`
}

func UploadData(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req DataUploadRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		// Create unique upload ID
		uploadID := primitive.NewObjectID()

		// Use absolute path for Docker volume
		dataDir := filepath.Join("/app/data/products", uploadID.Hex())

		// Ensure directory exists
		if err := os.MkdirAll(dataDir, 0755); err != nil {
			c.JSON(500, gin.H{"error": fmt.Sprintf("failed to create directory: %v", err)})
			return
		}

		// Save file
		filePath := filepath.Join(dataDir, req.FileName)
		if err := os.WriteFile(filePath, []byte(req.Content), 0644); err != nil {
			c.JSON(500, gin.H{"error": fmt.Sprintf("failed to write file: %v", err)})
			return
		}

		// Save metadata
		metadata := map[string]interface{}{
			"upload_id":  uploadID,
			"filename":   req.FileName,
			"filepath":   fmt.Sprintf("./data/products/%s/%s", uploadID.Hex(), req.FileName),
			"created_at": time.Now().UTC(),
		}

		if _, err := db.Collection("data_uploads").InsertOne(context.Background(), metadata); err != nil {
			c.JSON(500, gin.H{"error": fmt.Sprintf("failed to save metadata: %v", err)})
			return
		}

		c.JSON(200, gin.H{
			"upload_id": uploadID.Hex(),
			"location":  fmt.Sprintf("./data/products/%s/%s", uploadID.Hex(), req.FileName),
		})
	}
}
