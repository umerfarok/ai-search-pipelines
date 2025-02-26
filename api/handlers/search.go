package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/mongo"
)

type SearchService struct {
	db     *mongo.Database
	client *http.Client
}

func NewSearchService(db *mongo.Database) *SearchService {
	return &SearchService{
		db: db,
		client: &http.Client{
			Timeout: time.Second * 60,
		},
	}
}

func (s *SearchService) Search(c *gin.Context) {
	var searchReq struct {
		Query     string                 `json:"query" binding:"required"`
		ModelPath string                 `json:"model_path" binding:"required"`
		MaxItems  int                    `json:"max_items"`
		Filters   map[string]interface{} `json:"filters"`
		Page      int                    `json:"page"`
	}

	if err := c.ShouldBindJSON(&searchReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Default values
	if searchReq.MaxItems <= 0 {
		searchReq.MaxItems = 20
	}
	if searchReq.Page <= 0 {
		searchReq.Page = 1
	}

	// Prepare request to search service
	searchServiceHost := os.Getenv("SEARCH_SERVICE_HOST")
	if searchServiceHost == "" {
		searchServiceHost = "http://search-service:5001"
	}

	searchServiceURL := fmt.Sprintf("%s/search", searchServiceHost)

	// Forward request to search service
	requestBody, err := json.Marshal(map[string]interface{}{
		"query":      searchReq.Query,
		"model_path": searchReq.ModelPath,
		"max_items":  searchReq.MaxItems,
		"filters":    searchReq.Filters,
		"page":       searchReq.Page,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to marshal search request"})
		return
	}

	// Add more debug information
	log.Printf("Sending search request to: %s, Data: %s", searchServiceURL, string(requestBody))

	resp, err := s.client.Post(searchServiceURL, "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to call search service: %v", err)})
		return
	}
	defer resp.Body.Close()

	// Read and parse response
	responseBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read search response"})
		return
	}

	log.Printf("Search service response: %s", string(responseBody))

	// If the response status code is not 200, return the error
	if resp.StatusCode != http.StatusOK {
		var errorResp map[string]interface{}
		if err := json.Unmarshal(responseBody, &errorResp); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Search service error: %s", string(responseBody))})
			return
		}
		c.JSON(resp.StatusCode, errorResp)
		return
	}

	// Parse the response as JSON
	var searchResponse map[string]interface{}
	if err := json.Unmarshal(responseBody, &searchResponse); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse search response"})
		return
	}

	// Forward the response to the client
	c.JSON(http.StatusOK, searchResponse)
}

func (s *SearchService) Close() {
	// Any cleanup needed
}

func (s *SearchService) HealthCheck() bool {
	searchServiceHost := os.Getenv("SEARCH_SERVICE_HOST")
	if searchServiceHost == "" {
		searchServiceHost = "http://search-service:5001"
	}

	healthCheckURL := fmt.Sprintf("%s/health", searchServiceHost)
	log.Printf("Checking search service health at: %s", healthCheckURL)
	resp, err := s.client.Get(healthCheckURL)
	if err != nil {
		log.Printf("Search service health check failed: %v", err)
		return false
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	log.Printf("Health check response: %s", string(body))

	return resp.StatusCode == http.StatusOK
}
