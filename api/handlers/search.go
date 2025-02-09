package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
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

type SearchRequest struct {
	Query     string                 `json:"query"`
	ModelPath string                 `json:"model_path"`
	MaxItems  int                    `json:"max_items,omitempty"`
	Alpha     float64                `json:"alpha,omitempty"`
	Filters   map[string]interface{} `json:"filters,omitempty"`
	Page      int                    `json:"page,omitempty"`
}

type SearchResult struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Category    string                 `json:"category"`
	Score       float64                `json:"score"`
	Text        string                 `json:"text"`
	Metadata    map[string]interface{} `json:"metadata"`
	RerankScore float64                `json:"rerank_score,omitempty"`
}

type QueryInfo struct {
	Original  string                 `json:"original"`
	Enhanced  string                 `json:"enhanced"`
	Analysis  map[string]interface{} `json:"analysis"`
	ModelPath string                 `json:"model_path"`
}

type SearchResponse struct {
	Results      []SearchResult `json:"results,omitempty"`
	Total        int            `json:"total,omitempty"`
	QueryInfo    QueryInfo      `json:"query_info,omitempty"`
	Page         int            `json:"page,omitempty"`
	MaxItems     int            `json:"max_items,omitempty"`
	TextResponse string         `json:"text_response,omitempty"`
}

func (s *SearchService) Search(c *gin.Context) {
	var req SearchRequest

	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set default max items
	if req.MaxItems == 0 {
		req.MaxItems = 10
	}
	if req.Alpha == 0 {
		req.Alpha = 0.7
	}
	if req.Page == 0 {
		req.Page = 1
	}

	// If Query or ModelPath is empty, get the latest completed model
	if req.Query == "" || req.ModelPath == "" {
		modelConfig, err := s.getModelConfig("latest")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to get latest model config: %v", err)})
			return
		}
		req.ModelPath = modelConfig.ModelPath
	}

	// Forward request to search service
	jsonBody, err := json.Marshal(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to marshal request: %v", err)})
		return
	}

	// Call Python search service
	resp, err := http.Post(
		fmt.Sprintf("%s/search", s.searchHost),
		"application/json",
		bytes.NewBuffer(jsonBody),
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("search service error: %v", err)})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errorResp struct {
			Error string `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&errorResp); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "unknown search service error"})
			return
		}
		c.JSON(resp.StatusCode, gin.H{"error": errorResp.Error})
		return
	}

	// Read the response body
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to read response: %v", err)})
		return
	}

	// Try to decode as JSON first
	var searchResp SearchResponse
	if err := json.Unmarshal(responseBody, &searchResp); err != nil {
		// If JSON decoding fails, treat it as a text response
		searchResp = SearchResponse{
			TextResponse: string(responseBody),
			QueryInfo: QueryInfo{
				Original:  req.Query,
				ModelPath: req.ModelPath,
			},
			Page:     req.Page,
			MaxItems: req.MaxItems,
		}
	}

	// Apply filters if any and if we have structured results
	if len(req.Filters) > 0 && len(searchResp.Results) > 0 {
		searchResp.Results = s.applyFilters(searchResp.Results, req.Filters)
		searchResp.Total = len(searchResp.Results)
	}

	c.JSON(http.StatusOK, searchResp)
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

func (s *SearchService) applyFilters(results []SearchResult, filters map[string]interface{}) []SearchResult {
	filtered := make([]SearchResult, 0)

	for _, result := range results {
		if matchesFilters(result, filters) {
			filtered = append(filtered, result)
		}
	}

	return filtered
}

func matchesFilters(result SearchResult, filters map[string]interface{}) bool {
	for key, value := range filters {
		switch key {
		case "category":
			if categoryFilter, ok := value.(string); ok {
				if !matchesCategory(result.Category, categoryFilter) {
					return false
				}
			}
		case "price":
			if priceRange, ok := value.([]float64); ok && len(priceRange) == 2 {
				if price, exists := result.Metadata["price"].(float64); exists {
					if price < priceRange[0] || price > priceRange[1] {
						return false
					}
				}
			}
		default:
			// Check metadata fields
			if metaValue, exists := result.Metadata[key]; exists {
				if filterValue, ok := value.(string); ok {
					if !matchesMetadataField(metaValue, filterValue) {
						return false
					}
				}
			}
		}
	}

	return true
}

func matchesCategory(category, filter string) bool {
	// Implement category matching logic (e.g., exact match, hierarchical, fuzzy)
	return strings.Contains(strings.ToLower(category), strings.ToLower(filter))
}

func matchesMetadataField(value interface{}, filter string) bool {
	// Convert value to string for comparison
	strValue := fmt.Sprintf("%v", value)
	return strings.Contains(strings.ToLower(strValue), strings.ToLower(filter))
}

func (s *SearchService) Close() {
	close(s.healthCheck)
}
