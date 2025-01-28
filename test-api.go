package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

type SearchRequest struct {
	Query    string                 `json:"query"`
	ConfigID string                 `json:"config_id"`
	MaxItems int                    `json:"max_items"`
	Filters  map[string]interface{} `json:"filters"`
	Page     int                    `json:"page"`
}

func makeRequest(client *http.Client) bool {
	payload := SearchRequest{
		Query:    "i need something to clean my cat",
		ConfigID: "6799112e7f995b7ce3e5731c",
		MaxItems: 20,
		Filters:  map[string]interface{}{},
		Page:     1,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		log.Printf("Error marshaling JSON: %v", err)
		return false
	}

	req, err := http.NewRequest("POST", "http://localhost:8080/search", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Error creating request: %v", err)
		return false
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "*/*")
	req.Header.Set("Accept-Language", "en-GB,en-US;q=0.9,en;q=0.8")
	req.Header.Set("Referer", "http://localhost:3000/")

	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error making request: %v", err)
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == 200
}

func worker(id int, jobs <-chan int, results chan<- bool, client *http.Client) {
	for range jobs {
		success := makeRequest(client)
		results <- success
	}
}

func main() {
	numRequests := 1000
	numWorkers := 10

	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	jobs := make(chan int, numRequests)
	results := make(chan bool, numRequests)

	startTime := time.Now()

	var wg sync.WaitGroup
	for w := 1; w <= numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			worker(workerID, jobs, results, client)
		}(w)
	}

	for j := 1; j <= numRequests; j++ {
		jobs <- j
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	successCount := 0
	for success := range results {
		if success {
			successCount++
		}
	}

	elapsed := time.Since(startTime)
	fmt.Printf("Completed %d/%d requests successfully in %.2f seconds\n",
		successCount, numRequests, elapsed.Seconds())
	fmt.Printf("Success rate: %.2f%%\n", float64(successCount)/float64(numRequests)*100)
}
