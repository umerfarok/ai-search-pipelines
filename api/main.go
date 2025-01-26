package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/umerfarok/product-search/handlers"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// MongoDB setup
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

	// Initialize config service
	configService, err := handlers.NewConfigService(db)
	if err != nil {
		log.Fatal("Failed to initialize config service:", err)
	}

	// Setup Gin
	r := gin.Default()
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))


	r.POST("/config/create", configService.CreateConfigWithData)
	r.GET("/config/:id", handlers.GetConfig(db))
	r.GET("/config", handlers.ListConfigs(db))
	r.GET("/config/status/:id", configService.GetTrainingStatus)
	r.PUT("/config/status/:id", configService.UpdateModelVersionStatus)


	r.GET("/model/versions", handlers.GetModelVersions(db))
	r.GET("/model/version/:id", handlers.GetModelVersion(db))
	r.PUT("/model/version/:id/status", configService.UpdateModelVersionStatus)
	r.GET("/jobs/queue", configService.GetQueuedJobs)

	if searchHandler := handlers.NewSearchService(db); searchHandler != nil {
		r.POST("/search", searchHandler.Search)
	}


	r.GET("/training/status/:id", configService.GetTrainingStatus)

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting server on port %s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}
