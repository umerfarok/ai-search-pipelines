package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/mongo"
)

type SearchService struct {
	db             *mongo.Database
	searchEndpoint string
}

type SearchRequest struct {
	Query     string `json:"query"`
	ModelPath string `json:"model_path"`
	MaxItems  int    `json:"max_items"`
}

type SearchResponse struct {
	Results         []SearchResult `json:"results"`
	Total           int            `json:"total"`
	NaturalResponse string         `json:"natural_response"`
	QueryInfo       QueryInfo      `json:"query_info"`
}

type SearchResult struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Category    string            `json:"category"`
	Score       float64           `json:"score"`
	Metadata    map[string]string `json:"metadata"`
}

type QueryInfo struct {
	Original  string `json:"original"`
	ModelPath string `json:"model_path"`
}

func NewSearchService(db *mongo.Database) *SearchService {
	return &SearchService{
		db:             db,
		searchEndpoint: os.Getenv("SEARCH_SERVICE_URL"),
	}
}

func (s *SearchService) Search(c *gin.Context) {
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request format"})
		return
	}

	// Validate request
	if req.Query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query is required"})
		return
	}

	if req.ModelPath == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model_path is required"})
		return
	}

	// Set default max items if not provided
	if req.MaxItems == 0 {
		req.MaxItems = 10
	}

	// Forward request to search service
	searchResp, err := s.forwardSearchRequest(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("search service error: %v", err)})
		return
	}

	c.JSON(http.StatusOK, searchResp)
}

func (s *SearchService) forwardSearchRequest(req SearchRequest) (*SearchResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %v", err)
	}

	resp, err := http.Post(s.searchEndpoint+"/search", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error forwarding request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search service returned status: %d", resp.StatusCode)
	}

	var searchResp SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %v", err)
	}

	return &searchResp, nil
}

func extractScores(results []SearchResult) []float64 {
	scores := make([]float64, len(results))
	for i, result := range results {
		scores[i] = result.Score
	}
	return scores
}
