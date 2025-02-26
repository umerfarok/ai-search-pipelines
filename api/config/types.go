package config

import (
	"time"
)

type ModelStatus string

const (
	ModelStatusPending    ModelStatus = "pending"
	ModelStatusQueued     ModelStatus = "queued"
	ModelStatusProcessing ModelStatus = "processing"
	ModelStatusCompleted  ModelStatus = "completed"
	ModelStatusFailed     ModelStatus = "failed"
	ModelStatusCanceled   ModelStatus = "canceled"
)

// ModelConfig represents search model configuration settings
type ModelConfig struct {
	ID                string           `json:"id" bson:"_id,omitempty"`
	Name              string           `json:"name" bson:"name"`
	Description       string           `json:"description" bson:"description"`
	EmbeddingModel    string           `json:"embedding_model" bson:"embedding_model"`
	VectorSize        int              `json:"vector_size" bson:"vector_size"`
	CreatedAt         time.Time        `json:"created_at" bson:"created_at"`
	UpdatedAt         time.Time        `json:"updated_at" bson:"updated_at"`
	Status            string           `json:"status" bson:"status"`
	TrainingCompleted bool             `json:"training_completed" bson:"training_completed"`
	DataSourceConfig  DataSourceConfig `json:"data_source" bson:"data_source"`
	IndexConfig       IndexConfig      `json:"index_config" bson:"index_config"`
	SchemaMapping     SchemaMapping    `json:"schema_mapping" bson:"schema_mapping"`
	ModelPath         string           `json:"model_path" bson:"model_path"` // Add this field
}

// DataSourceConfig defines the data source for training
type DataSourceConfig struct {
	Type        string                 `json:"type" bson:"type"` // s3, postgres, csv, etc.
	Location    string                 `json:"location" bson:"location"`
	Credentials string                 `json:"credentials,omitempty" bson:"credentials,omitempty"`
	Format      string                 `json:"format" bson:"format"`
	Options     map[string]interface{} `json:"options" bson:"options"`
}

// IndexConfig defines vector index configuration
type IndexConfig struct {
	Type               string `json:"type" bson:"type"`                       // hnsw, flat, etc.
	DistanceMetric     string `json:"distance_metric" bson:"distance_metric"` // cosine, dot, euclidean
	HnswM              int    `json:"hnsw_m,omitempty" bson:"hnsw_m,omitempty"`
	HnswEfConstruction int    `json:"hnsw_ef_construction,omitempty" bson:"hnsw_ef_construction,omitempty"`
	HnswEfSearch       int    `json:"hnsw_ef_search,omitempty" bson:"hnsw_ef_search,omitempty"`
}

// SchemaMapping maps data fields to vector search fields
type SchemaMapping struct {
	IDColumn          string   `json:"idcolumn" bson:"idcolumn"`
	NameColumn        string   `json:"namecolumn" bson:"namecolumn"`
	DescriptionColumn string   `json:"descriptioncolumn" bson:"descriptioncolumn"`
	CategoryColumn    string   `json:"categorycolumn" bson:"categorycolumn"`
	CustomColumns     []Column `json:"customcolumns" bson:"customcolumns"`
	RequiredColumns   []string `json:"required_columns" bson:"required_columns"`
}

type Column struct {
	Name     string `json:"name" bson:"name"`
	Type     string `json:"type" bson:"type"`
	Role     string `json:"role" bson:"role"`
	Required bool   `json:"required" bson:"required"`
}

// ConfigQueue represents a training request in the queue
type ConfigQueue struct {
	ID          string     `json:"id" bson:"_id,omitempty"`
	ConfigID    string     `json:"config_id" bson:"config_id"`
	Status      string     `json:"status" bson:"status"` // pending, processing, completed, failed
	QueuedAt    time.Time  `json:"queued_at" bson:"queued_at"`
	ProcessedAt *time.Time `json:"processed_at,omitempty" bson:"processed_at,omitempty"`
	Error       string     `json:"error,omitempty" bson:"error,omitempty"`
}

type SearchRequest struct {
	Query    string                 `json:"query"`
	ConfigID string                 `json:"config_id"` // Use config ID instead of version
	MaxItems int                    `json:"max_items"`
	Filters  map[string]interface{} `json:"filters,omitempty"`
}

type SearchResult struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Category    string                 `json:"category"`
	Score       float64                `json:"score"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type SearchResponse struct {
	Results      []SearchResult `json:"results,omitempty"`
	Total        int            `json:"total,omitempty"`
	ConfigInfo   ModelConfig    `json:"config_info,omitempty"`
	TextResponse string         `json:"text_response,omitempty"`
}
type QueuedJob struct {
	ConfigID  string      `json:"config_id"`
	Config    ModelConfig `json:"config"`
	Timestamp string      `json:"timestamp"`
}

type LLMModel struct {
	Name        string `json:"name"`
	Path        string `json:"path"`
	Description string `json:"description"`
	Dimension   int    `json:"dimension"`
	IsDefault   bool   `json:"is_default"`
}

var AvailableLLMModels = map[string]LLMModel{
	"all-minilm-l6": {
		Name:        "All-MiniLM-L6",
		Path:        "sentence-transformers/all-MiniLM-L6-v2",
		Description: "Fast and efficient general-purpose embedding model",
		Dimension:   384,
		IsDefault:   true,
	},
	"bge-small": {
		Name:        "BGE Small",
		Path:        "BAAI/bge-small-en-v1.5",
		Description: "Small but effective embedding model",
		Dimension:   384,
		IsDefault:   false,
	},
	"bge-base": {
		Name:        "BGE Base",
		Path:        "BAAI/bge-base-en-v1.5",
		Description: "Medium-sized embedding model",
		Dimension:   768,
		IsDefault:   false,
	},
	"bge-large": {
		Name:        "BGE Large",
		Path:        "BAAI/bge-large-en-v1.5",
		Description: "Large, high-performance embedding model",
		Dimension:   1024,
		IsDefault:   false,
	},
}
