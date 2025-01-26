// api/config/types.go

package config

import (
	"time"

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

type ProductConfig struct {
	ID              primitive.ObjectID `json:"_id" bson:"_id"`
	SchemaMapping   SchemaMapping      `json:"schema_mapping" bson:"schema_mapping"`
	Name            string             `json:"name" bson:"name"` // Name of the model
	DataSource      DataSource         `json:"data_source" bson:"data_source"`
	Mode            string             `json:"mode" bson:"mode"` // "replace" or "append"
	TrainingConfig  TrainingConfig     `json:"training_config" bson:"training_config"`
	PreviousVersion string             `json:"previous_version,omitempty" bson:"previous_version,omitempty"`
	Status          string             `json:"status" bson:"status"`
	CreatedAt       string             `json:"created_at" bson:"created_at"`
	UpdatedAt       string             `json:"updated_at" bson:"updated_at"`
}
type DataSource struct {
	Type     string   `json:"type"`
	Location string   `json:"location"`
	Columns  []Column `json:"columns"`
}
type Column struct {
	Name string `json:"name"`
	Type string `json:"type"`
	Role string `json:"role"`
}
type TrainingConfig struct {
	ModelType      string `json:"model_type"`
	EmbeddingModel string `json:"embedding_model"`
	BatchSize      int    `json:"batch_size"`
	MaxTokens      int    `json:"max_tokens"`
}
type SchemaMapping struct {
	IDColumn          string         `json:"id_column"`
	NameColumn        string         `json:"name_column"`
	DescriptionColumn string         `json:"description_column"`
	CategoryColumn    string         `json:"category_column"`
	CustomColumns     []CustomColumn `json:"custom_columns"`
}

type CustomColumn struct {
	UserColumn     string `json:"user_column"`
	StandardColumn string `json:"standard_column"`
	Role           string `json:"role"`
}

type ModelVersion struct {
	ID        primitive.ObjectID `bson:"_id" json:"id"`
	ConfigID  primitive.ObjectID `bson:"config_id" json:"config_id"`
	Version   string             `bson:"version" json:"version"`
	Status    string             `bson:"status" json:"status"`
	S3Path    string             `bson:"s3_path" json:"s3_path"`
	CreatedAt string             `bson:"created_at" json:"created_at"`
	UpdatedAt string             `bson:"updated_at" json:"updated_at"`
	Error     string             `bson:"error,omitempty" json:"error,omitempty"`
	Config    interface{}        `bson:"config" json:"config"`
}
type Product struct {
	ID        primitive.ObjectID     `bson:"_id,omitempty" json:"id"`
	ConfigID  primitive.ObjectID     `bson:"config_id" json:"config_id"`
	Data      map[string]interface{} `bson:"data" json:"data"`
	CreatedAt time.Time              `bson:"created_at" json:"created_at"`
	UpdatedAt time.Time              `bson:"updated_at" json:"updated_at"`
	Status    string                 `bson:"status" json:"status"`
}

type SearchRequest struct {
	Query    string `json:"query"`
	Version  string `json:"version"`
	MaxItems int    `json:"max_items"`
}

type SearchResult struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Score       float64 `json:"score"`
	Category    string  `json:"category"`
}

type SearchResponse struct {
	Results []SearchResult `json:"results"`
	Total   int            `json:"total"`
}

type ModelConfig struct {
	ID             primitive.ObjectID `bson:"_id" json:"id"`
	Name           string             `bson:"name" json:"name"`
	Description    string             `bson:"description" json:"description"`
	DataSource     DataSource         `bson:"data_source" json:"data_source"`
	SchemaMapping  SchemaMapping      `bson:"schema_mapping" json:"schema_mapping"`
	TrainingConfig TrainingConfig     `bson:"training_config" json:"training_config"`
	CreatedAt      time.Time          `bson:"created_at" json:"created_at"`
	UpdatedAt      time.Time          `bson:"updated_at" json:"updated_at"`
	Status         string             `bson:"status" json:"status"`
}

type QueuedJob struct {
	VersionID string                 `json:"version_id"`
	ConfigID  string                 `json:"config_id"`
	Config    map[string]interface{} `json:"config"`
	S3Path    string                 `json:"s3_path"`
}