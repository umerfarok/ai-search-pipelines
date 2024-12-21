// api/handlers/config.go

package handlers

import (
	"context"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/umerfarok/product-search/config"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

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

func CreateConfig(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		var cfg config.ProductConfig
		if err := c.BindJSON(&cfg); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		// Validate required fields
		if cfg.Name == "" || cfg.DataSource.Type == "" || cfg.DataSource.Location == "" {
			c.JSON(400, gin.H{"error": "name, data_source.type, and data_source.location are required"})
			return
		}

		cfg.ID = primitive.NewObjectID()
		cfg.CreatedAt = time.Now().UTC()
		cfg.UpdatedAt = cfg.CreatedAt
		cfg.Status = "created"

		_, err := db.Collection("configs").InsertOne(context.Background(), cfg)
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to create configuration: " + err.Error()})
			return
		}

		c.JSON(201, cfg)
	}
}

func GetConfig(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		id, err := primitive.ObjectIDFromHex(c.Param("id"))
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid id format"})
			return
		}

		var cfg config.ProductConfig
		err = db.Collection("configs").FindOne(context.Background(), bson.M{"_id": id}).Decode(&cfg)
		if err != nil {
			if err == mongo.ErrNoDocuments {
				c.JSON(404, gin.H{"error": "configuration not found"})
				return
			}
			c.JSON(500, gin.H{"error": "failed to fetch configuration: " + err.Error()})
			return
		}

		c.JSON(200, cfg)
	}
}

func UpdateConfig(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		id, err := primitive.ObjectIDFromHex(c.Param("id"))
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid id format"})
			return
		}

		var cfg config.ProductConfig
		if err := c.BindJSON(&cfg); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		cfg.ID = id
		cfg.UpdatedAt = time.Now().UTC()

		result, err := db.Collection("configs").UpdateOne(
			context.Background(),
			bson.M{"_id": id},
			bson.M{"$set": cfg},
		)

		if err != nil {
			c.JSON(500, gin.H{"error": "failed to update configuration: " + err.Error()})
			return
		}

		if result.MatchedCount == 0 {
			c.JSON(404, gin.H{"error": "configuration not found"})
			return
		}

		c.JSON(200, cfg)
	}
}

func ListConfigs(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		filter := bson.M{}

		// Add status filter if provided
		if status := c.Query("status"); status != "" {
			filter["status"] = status
		}

		cursor, err := db.Collection("configs").Find(context.Background(), filter)
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to fetch configurations: " + err.Error()})
			return
		}
		defer cursor.Close(context.Background())

		var configs []config.ProductConfig
		if err := cursor.All(context.Background(), &configs); err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}

		c.JSON(200, configs)
	}
}

func DeleteConfig(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		id, err := primitive.ObjectIDFromHex(c.Param("id"))
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid id format"})
			return
		}

		result, err := db.Collection("configs").DeleteOne(context.Background(), bson.M{"_id": id})
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to delete configuration: " + err.Error()})
			return
		}

		if result.DeletedCount == 0 {
			c.JSON(404, gin.H{"error": "configuration not found"})
			return
		}

		c.JSON(200, gin.H{"message": "configuration deleted successfully"})
	}
}
