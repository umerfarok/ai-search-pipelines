// api/handlers/search.go

package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/umerfarok/product-search/config"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type SearchService struct {
	db          *mongo.Database
	searchHost  string
	cache       sync.Map
	healthCheck chan struct{}
}

func NewSearchService(db *mongo.Database) *SearchService {
	searchHost := os.Getenv("SEARCH_SERVICE_HOST")
	if searchHost == "" {
		searchHost = "http://localhost:5001"
	}

	s := &SearchService{
		db:          db,
		searchHost:  searchHost,
		healthCheck: make(chan struct{}),
	}

	// Start health check routine
	go s.startHealthCheck()

	return s
}

func (s *SearchService) startHealthCheck() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.checkHealth()
		case <-s.healthCheck:
			return
		}
	}
}

func (s *SearchService) checkHealth() {
	resp, err := http.Get(fmt.Sprintf("%s/health", s.searchHost))
	if err != nil || resp.StatusCode != http.StatusOK {
		log.Printf("Search service health check failed: %v", err)
		return
	}
	defer resp.Body.Close()
}

func (s *SearchService) Search(c *gin.Context) {
	var req config.SearchRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	if req.Query == "" {
		c.JSON(400, gin.H{"error": "query is required"})
		return
	}

	if req.MaxItems <= 0 {
		req.MaxItems = 10 // Default value
	}

	// Get model version
	version, err := s.getModelVersion(req.Version)
	if err != nil {
		c.JSON(404, gin.H{"error": err.Error()})
		return
	}

	// Perform search
	results, err := s.performSearch(version, req)
	if err != nil {
		c.JSON(500, gin.H{"error": "search failed: " + err.Error()})
		return
	}

	c.JSON(200, results)
}

func (s *SearchService) getModelVersion(versionID string) (config.ModelVersion, error) {
	var version config.ModelVersion

	if versionID == "latest" {
		opts := options.FindOne().SetSort(bson.D{{Key: "created_at", Value: -1}})
		err := s.db.Collection("model_versions").FindOne(
			context.Background(),
			bson.M{"status": "completed"},
			opts,
		).Decode(&version)
		if err != nil {
			return version, fmt.Errorf("no completed model found")
		}
	} else {
		id, err := primitive.ObjectIDFromHex(versionID)
		if err != nil {
			return version, fmt.Errorf("invalid version format")
		}

		err = s.db.Collection("model_versions").FindOne(
			context.Background(),
			bson.M{"_id": id},
		).Decode(&version)
		if err != nil {
			return version, fmt.Errorf("model version not found")
		}
	}

	return version, nil
}

// api/handlers/search.go

func (s *SearchService) performSearch(version config.ModelVersion, req config.SearchRequest) (*config.SearchResponse, error) {
	// Ensure the model path is absolute and clean
	modelPath := filepath.Clean(version.ArtifactPath)
	if !filepath.IsAbs(modelPath) {
		modelPath = filepath.Join("./models", version.Version)
	}

	requestBody := map[string]interface{}{
		"model_path": modelPath,
		"query":      req.Query,
		"max_items":  req.MaxItems,
	}

	jsonBody, err := json.Marshal(requestBody)
	log.Printf("Sending search request with model path: %s", modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	log.Printf("Sending search request with model path: %s", modelPath)

	resp, err := http.Post(fmt.Sprintf("%s/search", s.searchHost), "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to call search service: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("search service error: %s", string(body))
	}

	var searchResp config.SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	return &searchResp, nil
}
func (s *SearchService) Close() {
	close(s.healthCheck)
}
