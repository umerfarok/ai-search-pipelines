// api/handlers/search.go

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
	var req struct {
		Query    string `json:"query"`
		Version  string `json:"version,omitempty"`
		MaxItems int    `json:"max_items,omitempty"`
	}

	if err := c.BindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Get latest version if not specified
	var modelVersion config.ModelVersion
	var err error

	if req.Version == "latest" || req.Version == "" {
		err = s.db.Collection("model_versions").FindOne(
			context.Background(),
			bson.M{"status": "completed"},
			options.FindOne().SetSort(bson.D{{Key: "created_at", Value: -1}}),
		).Decode(&modelVersion)
	} else {
		objID, err := primitive.ObjectIDFromHex(req.Version)
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid version id"})
			return
		}
		err = s.db.Collection("model_versions").FindOne(
			context.Background(),
			bson.M{"_id": objID},
		).Decode(&modelVersion)
	}

	if err != nil {
		c.JSON(404, gin.H{"error": "model version not found"})
		return
	}

	// Forward search request to search service
	searchReq := struct {
		Query     string `json:"query"`
		ModelPath string `json:"model_path"`
		MaxItems  int    `json:"max_items"`
	}{
		Query:     req.Query,
		ModelPath: modelVersion.S3Path,
		MaxItems:  req.MaxItems,
	}

	jsonBody, err := json.Marshal(searchReq)
	if err != nil {
		c.JSON(500, gin.H{"error": "failed to marshal search request: " + err.Error()})
		return
	}

	resp, err := http.Post(
		fmt.Sprintf("%s/search", s.searchHost),
		"application/json",
		bytes.NewBuffer(jsonBody),
	)
	if err != nil {
		c.JSON(500, gin.H{"error": "search service error: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	var searchResults interface{}
	if err := json.NewDecoder(resp.Body).Decode(&searchResults); err != nil {
		c.JSON(500, gin.H{"error": "failed to decode search results"})
		return
	}

	c.JSON(200, searchResults)
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

// func (s *SearchService) performSearch(version config.ModelVersion, req config.SearchRequest) (*config.SearchResponse, error) {
// 	if version.S3Path == "" {
// 		return nil, fmt.Errorf("model version has no S3 path configured")
// 	}

// 	requestBody := map[string]interface{}{
// 		"model_path": version.S3Path,
// 		"query":      req.Query,
// 		"max_items":  req.MaxItems,
// 	}

// 	jsonBody, err := json.Marshal(requestBody)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to marshal request: %v", err)
// 	}

// 	log.Printf("Sending search request with model path: %s", version.S3Path)

// 	resp, err := http.Post(fmt.Sprintf("%s/search", s.searchHost), "application/json", bytes.NewBuffer(jsonBody))
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to call search service: %v", err)
// 	}
// 	defer resp.Body.Close()

// 	if resp.StatusCode != http.StatusOK {
// 		body, _ := ioutil.ReadAll(resp.Body)
// 		return nil, fmt.Errorf("search service error: %s", string(body))
// 	}

// 	var searchResp config.SearchResponse
// 	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
// 		return nil, fmt.Errorf("failed to decode response: %v", err)
// 	}

//		return &searchResp, nil
//	}
func (s *SearchService) Close() {
	close(s.healthCheck)
}
