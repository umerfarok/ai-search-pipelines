// main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/umerfarok/product-search/handlers"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	mongoURI := os.Getenv("MONGO_URI")
	if mongoURI == "" {
		mongoURI = "mongodb://root:example@localhost:27017"
	}

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoURI))
	if err != nil {
		log.Fatal("Failed to connect to MongoDB:", err)
	}
	defer client.Disconnect(ctx)

	if err := client.Ping(ctx, nil); err != nil {
		log.Fatal("Failed to ping MongoDB:", err)
	}

	db := client.Database("product_search")

	if err := createIndexes(db); err != nil {
		log.Fatal("Failed to create indexes:", err)
	}

	// Initialize services
	searchService := handlers.NewSearchService(db)
	trainingService := handlers.NewTrainingService(db)

	r := gin.Default()

	// Config routes
	r.POST("/config", handlers.CreateConfig(db))
	r.GET("/config/:id", handlers.GetConfig(db))
	r.GET("/config", handlers.ListConfigs(db))

	// Training routes
	r.POST("/model/train", trainingService.TriggerTraining)
	r.GET("/model/versions", handlers.GetModelVersions(db))
	r.GET("/model/version/:id", handlers.GetModelVersion(db))
	r.POST("/data/upload", handlers.UploadData(db))
	// Product routes
	r.POST("/products/update", handlers.UpdateProducts(db))
	r.GET("/products", handlers.GetProducts(db))

	// Search routes
	r.POST("/search", searchService.Search)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting server on port %s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

func createIndexes(db *mongo.Database) error {
	collections := map[string][]mongo.IndexModel{
		"configs": {
			{Keys: bson.D{{Key: "created_at", Value: -1}}},
			{Keys: bson.D{{Key: "status", Value: 1}}},
		},
		"model_versions": {
			{Keys: bson.D{{Key: "config_id", Value: 1}}},
			{Keys: bson.D{{Key: "created_at", Value: -1}}},
			{Keys: bson.D{{Key: "status", Value: 1}}},
		},
		"products": {
			{Keys: bson.D{{Key: "config_id", Value: 1}}},
			{Keys: bson.D{{Key: "created_at", Value: -1}}},
			{Keys: bson.D{{Key: "status", Value: 1}}},
		},
	}

	for collection, indexes := range collections {
		_, err := db.Collection(collection).Indexes().CreateMany(context.Background(), indexes)
		if err != nil {
			return fmt.Errorf("failed to create indexes for %s: %v", collection, err)
		}
	}

	return nil
}
