// api/config/types.go

package config

import (
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

type ProductConfig struct {
	ID             primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Name           string             `bson:"name" json:"name"`
	Description    string             `bson:"description" json:"description"`
	DataSource     DataSource         `bson:"data_source" json:"data_source"`
	TrainingConfig TrainingConfig     `bson:"training_config" json:"training_config"`
	SchemaMapping  SchemaMapping      `bson:"schema_mapping" json:"schema_mapping"`
	CreatedAt      time.Time          `bson:"created_at" json:"created_at"`
	UpdatedAt      time.Time          `bson:"updated_at" json:"updated_at"`
	Status         string             `bson:"status" json:"status"`
}

type DataSource struct {
	Type      string   `bson:"type" json:"type"`
	Location  string   `bson:"location" json:"location"`
	TableName string   `bson:"table_name,omitempty" json:"table_name,omitempty"`
	Query     string   `bson:"query,omitempty" json:"query,omitempty"`
	Columns   []Column `bson:"columns" json:"columns"`
}

type Column struct {
	Name        string `bson:"name" json:"name"`
	Type        string `bson:"type" json:"type"`
	Role        string `bson:"role" json:"role"`
	Description string `bson:"description" json:"description"`
}

type TrainingConfig struct {
	ModelType      string `bson:"model_type" json:"model_type"`
	EmbeddingModel string `bson:"embedding_model" json:"embedding_model"`
	ZeroShotModel  string `bson:"zero_shot_model" json:"zero_shot_model"`
	BatchSize      int    `bson:"batch_size" json:"batch_size"`
	MaxTokens      int    `bson:"max_tokens" json:"max_tokens"`
}

type SchemaMapping struct {
	IDColumn       string         `bson:"id_column" json:"id_column"`
	NameColumn     string         `bson:"name_column" json:"name_column"`
	DescColumn     string         `bson:"description_column" json:"description_column"`
	CategoryColumn string         `bson:"category_column,omitempty" json:"category_column,omitempty"`
	PriceColumn    string         `bson:"price_column,omitempty" json:"price_column,omitempty"`
	CustomColumns  []CustomColumn `bson:"custom_columns,omitempty" json:"custom_columns,omitempty"`
}

type CustomColumn struct {
	UserColumn     string `bson:"user_column" json:"user_column"`
	StandardColumn string `bson:"standard_column" json:"standard_column"`
	Role           string `bson:"role" json:"role"`
}

type ModelVersion struct {
	ID           primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	ConfigID     primitive.ObjectID `bson:"config_id" json:"config_id"`
	Version      string             `bson:"version" json:"version"`
	Status       string             `bson:"status" json:"status"`
	CreatedAt    time.Time          `bson:"created_at" json:"created_at"`
	ArtifactPath string             `bson:"artifact_path" json:"artifact_path"`
	Error        string             `bson:"error,omitempty" json:"error,omitempty"`
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
