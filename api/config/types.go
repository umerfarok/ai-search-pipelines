package config

import (
	"go.mongodb.org/mongo-driver/bson/primitive"
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

type ModelConfig struct {
	ID              primitive.ObjectID `json:"_id" bson:"_id"`
	Name            string             `json:"name" bson:"name"`
	Description     string             `json:"description" bson:"description"`
	ModelPath       string             `json:"model_path" bson:"model_path"`
	DataSource      DataSource         `json:"data_source" bson:"data_source"`
	SchemaMapping   SchemaMapping      `json:"schema_mapping" bson:"schema_mapping"`
	TrainingConfig  TrainingConfig     `json:"training_config" bson:"training_config"`
	Status          string             `json:"status" bson:"status"`
	Error           string             `json:"error,omitempty" bson:"error,omitempty"`
	Mode            string             `json:"mode" bson:"mode"` // "replace" or "append"
	PreviousVersion string             `json:"previous_version,omitempty" bson:"previous_version,omitempty"`
	Version         string             `json:"version" bson:"version"` // Timestamp-based version identifier
	CreatedAt       string             `json:"created_at" bson:"created_at"`
	UpdatedAt       string             `json:"updated_at" bson:"updated_at"`
	TrainingStats   *TrainingStats     `json:"training_stats,omitempty" bson:"training_stats,omitempty"`
}

type DataSource struct {
	Type         string   `json:"type"`
	Location     string   `json:"location"`
	FileType     string   `json:"file_type"` // e.g., "csv", "json"
	Columns      []Column `json:"columns"`
	TotalRecords int      `json:"total_records,omitempty"`
}

type Column struct {
	Name     string `json:"name"`
	Type     string `json:"type"`
	Role     string `json:"role"`
	Required bool   `json:"required"`
}

type TrainingConfig struct {
	ModelType       string            `json:"model_type"`
	EmbeddingModel  string            `json:"embeddingmodel"`
	BatchSize       int               `json:"batch_size"`
	MaxTokens       int               `json:"max_tokens"`
	LLMModel        string            `json:"llm_model"` // Add this field for LLM model selection
	TrainingParams  map[string]string `json:"training_params,omitempty"`
	ValidationSplit float64           `json:"validation_split"`
}

type SchemaMapping struct {
	Idcolumn          string         `json:"idcolumn" bson:"idcolumn"`
	Namecolumn        string         `json:"namecolumn" bson:"namecolumn"`
	Descriptioncolumn string         `json:"descriptioncolumn" bson:"descriptioncolumn"`
	Categorycolumn    string         `json:"categorycolumn" bson:"categorycolumn"`
	Customcolumns     []CustomColumn `json:"customcolumns" bson:"customcolumns"` // Update to use CustomColumn struct
}

type CustomColumn struct {
	Name     string `json:"name" bson:"name"`
	Type     string `json:"type" bson:"type"` // text, number, category, url
	Role     string `json:"role" bson:"role"` // training, metadata
	Required bool   `json:"required" bson:"required"`
}

type TrainingStats struct {
	StartTime        string  `json:"start_time"`
	EndTime          string  `json:"end_time,omitempty"`
	ProcessedRecords int     `json:"processed_records"`
	TotalRecords     int     `json:"total_records"`
	TrainingAccuracy float64 `json:"training_accuracy,omitempty"`
	ValidationScore  float64 `json:"validation_score,omitempty"`
	ErrorRate        float64 `json:"error_rate,omitempty"`
	Progress         float64 `json:"progress"`
	LLMModel         string  `json:"llm_model,omitempty"`      // Add this field
	LLMModelName     string  `json:"llm_model_name,omitempty"` // Add this field
	CompletedAt      string  `json:"completed_at,omitempty"`   // Add this field
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

// Optional: Add a type for available LLM models (can be used for validation)
type LLMModelInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

var AvailableLLMModels = map[string]LLMModelInfo{
	"deepseek-coder": {
		Name:        "DeepSeek Coder 1.3B",
		Description: "Code-optimized model for technical content",
	},
	"gpt2": {
		Name:        "GPT-2",
		Description: "General purpose language model",
	},
	"opt-350m": {
		Name:        "OPT 350M",
		Description: "Compact language model for general text",
	},
}
