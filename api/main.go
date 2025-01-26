package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/umerfarok/product-search/handlers"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func initMongoDB() (*mongo.Database, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Get MongoDB URI from environment
	mongoURI := os.Getenv("MONGO_URI")
	if mongoURI == "" {
		mongoURI = "mongodb://localhost:27017"
	}

	// Connect to MongoDB
	clientOptions := options.Client().ApplyURI(mongoURI)
	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		return nil, err
	}

	// Ping the database
	if err := client.Ping(ctx, nil); err != nil {
		return nil, err
	}

	dbName := os.Getenv("MONGO_DB_NAME")
	if dbName == "" {
		dbName = "product_search"
	}

	return client.Database(dbName), nil
}

func setupCORS() gin.HandlerFunc {
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"*"} // Configure based on your needs
	config.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{
		"Origin",
		"Content-Type",
		"Accept",
		"Authorization",
		"X-Requested-With",
	}
	config.ExposeHeaders = []string{"Content-Length"}
	config.AllowCredentials = true
	config.MaxAge = 12 * time.Hour

	return cors.New(config)
}

func setupRouter(configService *handlers.ConfigService, searchService *handlers.SearchService) *gin.Engine {
	r := gin.Default()

	// Add CORS middleware
	r.Use(setupCORS())

	// Add recovery middleware
	r.Use(gin.Recovery())

	// Health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status": "ok",
			"time":   time.Now().UTC(),
		})
	})

	// Config endpoints
	r.POST("/config", configService.CreateConfig)
	r.GET("/config/:id", configService.GetConfig)
	r.GET("/config", configService.ListConfigs)
	r.GET("/config/status/:id", configService.GetTrainingStatus)
	r.PUT("/config/status/:id", configService.UpdateConfigStatus)

	// Queue monitoring
	r.GET("/queue", configService.GetQueuedJobs)

	// Search endpoints
	if searchService != nil {
		r.POST("/search", searchService.Search)
	}

	return r
}

func main() {
	// Load .env file if it exists
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found")
	}

	// Set Gin mode
	ginMode := os.Getenv("GIN_MODE")
	if ginMode == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Initialize MongoDB
	db, err := initMongoDB()
	if err != nil {
		log.Fatalf("Failed to connect to MongoDB: %v", err)
	}

	// Initialize config service
	configService, err := handlers.NewConfigService(db)
	if err != nil {
		log.Fatalf("Failed to initialize config service: %v", err)
	}

	// Initialize search service
	searchService := handlers.NewSearchService(db)

	// Setup router
	router := setupRouter(configService, searchService)

	// Get port from environment
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Create server
	srv := &http.Server{
		Addr:    ":" + port,
		Handler: router,
	}

	// Start server in a goroutine
	go func() {
		log.Printf("Server starting on port %s", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	// Create shutdown context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Cleanup services
	if searchService != nil {
		searchService.Close()
	}

	// Shutdown server
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("Server exited gracefully")
}
