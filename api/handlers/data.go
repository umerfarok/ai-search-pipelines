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
	Filename string `json:"filename"`
	Content  string `json:"content"`
}

func UploadData(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req DataUploadRequest
		if err := c.BindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		// Create unique directory for this upload
		dirID := primitive.NewObjectID()
		dataDir := filepath.Join("data", "products", dirID.Hex())

		if err := os.MkdirAll(dataDir, 0755); err != nil {
			c.JSON(500, gin.H{"error": "failed to create data directory"})
			return
		}

		// Save file
		filePath := filepath.Join(dataDir, req.Filename)
		if err := os.WriteFile(filePath, []byte(req.Content), 0644); err != nil {
			c.JSON(500, gin.H{"error": "failed to write file"})
			return
		}

		// Save metadata
		metadata := map[string]interface{}{
			"filepath":  filePath,
			"filename":  req.Filename,
			"uploaded":  time.Now(),
			"upload_id": dirID,
		}

		_, err := db.Collection("data_uploads").InsertOne(context.Background(), metadata)
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to save metadata"})
			return
		}

		c.JSON(200, gin.H{
			"filepath":  fmt.Sprintf("./data/products/%s/%s", dirID.Hex(), req.Filename),
			"upload_id": dirID.Hex(),
		})
	}
}
