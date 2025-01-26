package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
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
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set default max items if not specified
	if req.MaxItems == 0 {
		req.MaxItems = 10
	}

	// Get the appropriate model configuration
	modelConfig, err := s.getModelConfig(req.ConfigID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("failed to get model config: %v", err)})
		return
	}

	// Ensure model is in a completed state
	if modelConfig.Status != string(config.ModelStatusCompleted) {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("model is not ready for search (status: %s)", modelConfig.Status),
		})
		return
	}

	// Forward search request to search service
	searchReq := struct {
		Query     string                 `json:"query"`
		ModelPath string                 `json:"model_path"`
		MaxItems  int                    `json:"max_items"`
		Filters   map[string]interface{} `json:"filters,omitempty"`
	}{
		Query:     req.Query,
		ModelPath: modelConfig.ModelPath,
		MaxItems:  req.MaxItems,
		Filters:   req.Filters,
	}

	response, err := s.performSearch(searchReq, modelConfig)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("search failed: %v", err)})
		return
	}

	c.JSON(http.StatusOK, response)
}

func (s *SearchService) getModelConfig(configID string) (*config.ModelConfig, error) {
	var modelConfig config.ModelConfig

	if configID == "latest" {
		// Get the latest completed model
		opts := options.FindOne().SetSort(bson.D{{Key: "created_at", Value: -1}})
		err := s.db.Collection("configs").FindOne(
			context.Background(),
			bson.M{"status": string(config.ModelStatusCompleted)},
			opts,
		).Decode(&modelConfig)
		if err != nil {
			return nil, fmt.Errorf("no completed models found")
		}
	} else {
		// Get specific model by ID
		objID, err := primitive.ObjectIDFromHex(configID)
		if err != nil {
			return nil, fmt.Errorf("invalid config ID format")
		}

		err = s.db.Collection("configs").FindOne(
			context.Background(),
			bson.M{"_id": objID},
		).Decode(&modelConfig)
		if err != nil {
			return nil, fmt.Errorf("model config not found")
		}
	}

	return &modelConfig, nil
}

func (s *SearchService) performSearch(req interface{}, modelConfig *config.ModelConfig) (*config.SearchResponse, error) {
	if modelConfig.ModelPath == "" {
		return nil, fmt.Errorf("model path not configured")
	}

	jsonBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Perform the search request
	resp, err := http.Post(
		fmt.Sprintf("%s/search", s.searchHost),
		"application/json",
		bytes.NewBuffer(jsonBody),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call search service: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errorResponse struct {
			Error string `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&errorResponse); err != nil {
			return nil, fmt.Errorf("search service error (status %d)", resp.StatusCode)
		}
		return nil, fmt.Errorf("search service error: %s", errorResponse.Error)
	}

	var searchResp config.SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	// Add config info to response
	searchResp.ConfigInfo = *modelConfig

	return &searchResp, nil
}

func (s *SearchService) enhanceSearchResults(results []config.SearchResult, modelConfig *config.ModelConfig) []config.SearchResult {
	// Add any additional metadata or post-processing of search results
	for i := range results {
		// Example: Add schema information or other relevant metadata
		results[i].Metadata = map[string]interface{}{
			"schema_version": modelConfig.Version,
			"model_name":     modelConfig.Name,
		}
	}
	return results
}

func (s *SearchService) Close() {
	close(s.healthCheck)
}
