// api/handlers/product.go

package handlers

import (
	"context"
	"encoding/csv"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/umerfarok/product-search/config"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func UpdateProducts(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			ConfigID   string `json:"config_id"`
			Mode       string `json:"mode"` // append or replace
			CSVContent string `json:"csv_content"`
		}
		if err := c.BindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		if req.Mode != "append" && req.Mode != "replace" {
			c.JSON(400, gin.H{"error": "mode must be either 'append' or 'replace'"})
			return
		}

		configID, err := primitive.ObjectIDFromHex(req.ConfigID)
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid config_id format"})
			return
		}

		// Get config
		var cfg config.ProductConfig
		if err := db.Collection("configs").FindOne(context.Background(), bson.M{"_id": configID}).Decode(&cfg); err != nil {
			c.JSON(404, gin.H{"error": "configuration not found"})
			return
		}

		// Parse CSV content
		reader := csv.NewReader(strings.NewReader(req.CSVContent))
		records, err := reader.ReadAll()
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid CSV content"})
			return
		}

		if len(records) < 2 {
			c.JSON(400, gin.H{"error": "CSV must contain headers and at least one row"})
			return
		}
		headers := records[0]

		// Start session
		session, err := db.Client().StartSession()
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to start session"})
			return
		}

		err = session.StartTransaction()
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to start transaction"})
			return
		}

		// If replace mode, delete existing products
		if req.Mode == "replace" {
			_, err = db.Collection("products").DeleteMany(
				context.Background(),
				bson.M{"config_id": configID},
			)
			if err != nil {
				session.AbortTransaction(context.Background())
				c.JSON(500, gin.H{"error": "failed to delete existing products"})
				return
			}
		}

		// Process products
		products := make([]interface{}, 0)
		for _, record := range records[1:] {
			data := make(map[string]interface{})
			for i, value := range record {
				if i < len(headers) {
					data[headers[i]] = value
				}
			}

			product := config.Product{
				ID:        primitive.NewObjectID(),
				ConfigID:  configID,
				Data:      data,
				CreatedAt: time.Now().UTC(),
				UpdatedAt: time.Now().UTC(),
				Status:    "active",
			}
			products = append(products, product)
		}

		// Save products to database
		if len(products) > 0 {
			_, err := db.Collection("products").InsertMany(context.Background(), products)
			if err != nil {
				session.AbortTransaction(context.Background())
				c.JSON(500, gin.H{"error": "failed to save products"})
				return
			}
		}

		// Save CSV file for training
		dataDir := filepath.Join("data", "products", req.ConfigID)
		if err := os.MkdirAll(dataDir, 0755); err != nil {
			session.AbortTransaction(context.Background())
			c.JSON(500, gin.H{"error": "failed to create data directory"})
			return
		}

		csvPath := filepath.Join(dataDir, "products.csv")
		f, err := os.Create(csvPath)
		if err != nil {
			session.AbortTransaction(context.Background())
			c.JSON(500, gin.H{"error": "failed to create CSV file"})
			return
		}
		defer func() {
			if err := f.Close(); err != nil {
				c.JSON(500, gin.H{"error": "failed to close CSV file"})
			}
		}()

		if _, err := f.WriteString(req.CSVContent); err != nil {
			session.AbortTransaction(context.Background())
			c.JSON(500, gin.H{"error": "failed to write CSV file"})
			return
		}

		// Update config with file location
		update := bson.M{
			"$set": bson.M{
				"data_source.location": csvPath,
				"updated_at":           time.Now().UTC(),
			},
		}

		_, err = db.Collection("configs").UpdateOne(
			context.Background(),
			bson.M{"_id": configID},
			update,
		)

		if err != nil {
			session.AbortTransaction(context.Background())
			c.JSON(500, gin.H{"error": "failed to update config"})
			return
		}

		// Commit transaction
		err = session.CommitTransaction(context.Background())
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to commit transaction"})
			return
		}

		c.JSON(200, gin.H{
			"message": "Products updated successfully",
			"count":   len(products),
		})
	}
}

func GetProducts(db *mongo.Database) gin.HandlerFunc {
	return func(c *gin.Context) {
		configID, err := primitive.ObjectIDFromHex(c.Query("config_id"))
		if err != nil {
			c.JSON(400, gin.H{"error": "invalid config_id format"})
			return
		}

		filter := bson.M{"config_id": configID}
		if status := c.Query("status"); status != "" {
			filter["status"] = status
		}

		opts := options.Find()
		if limit := c.Query("limit"); limit != "" {
			limitInt, err := strconv.Atoi(limit)
			if err == nil && limitInt > 0 {
				opts.SetLimit(int64(limitInt))
			}
		}

		cursor, err := db.Collection("products").Find(context.Background(), filter, opts)
		if err != nil {
			c.JSON(500, gin.H{"error": "failed to fetch products"})
			return
		}
		defer cursor.Close(context.Background())

		var products []config.Product
		if err := cursor.All(context.Background(), &products); err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}

		c.JSON(200, products)
	}
}
