const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const FormData = require('form-data');
const { createReadStream } = require('fs');

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8080';
const TRAINING_TIMEOUT = 30 * 60 * 1000; // 30 minutes
const CHECK_INTERVAL = 5000; // 5 seconds

// Test configuration matching our new schema
const TEST_CONFIG = {
    name: 'Appliance Product Search',
    description: 'Product search configuration for appliances dataset',
    mode: 'replace', // or 'append'
    data_source: {
        type: 'csv',
        file_type: 'csv',
        columns: [], // Will be populated after CSV analysis
    },
    schema_mapping: {
        id_column: 'SKU',
        name_column: 'PRODUCT_NAME',
        description_column: 'BREADCRUMBS',
        category_column: 'CATEGORY',
        custom_columns: [
            {
                user_column: 'BRAND',
                standard_column: 'brand',
                role: 'training',
                required: true
            },
            {
                user_column: 'SUBCATEGORY',
                standard_column: 'subcategory',
                role: 'training',
                required: false
            },
            {
                user_column: 'PRICE_CURRENT',
                standard_column: 'price',
                role: 'metadata',
                required: false
            },
            {
                user_column: 'SELLER',
                standard_column: 'seller',
                role: 'metadata',
                required: false
            }
        ],
        required_columns: ['SKU', 'PRODUCT_NAME', 'CATEGORY']
    },
    training_config: {
        model_type: 'transformer',
        embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
        batch_size: 128,
        max_tokens: 512,
        validation_split: 0.2,
        training_params: {}
    }
};

async function waitForService(url, maxAttempts = 30) {
    console.log('Waiting for service to be ready...');
    for (let i = 0; i < maxAttempts; i++) {
        try {
            await axios.get(`${url}/health`);
            console.log('✓ Service is ready');
            return true;
        } catch (error) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    throw new Error('Service failed to become ready');
}

async function createConfiguration(csvPath) {
    try {
        console.log('Creating new configuration...');
        
        // Create form data with file and config
        const formData = new FormData();
        formData.append('file', createReadStream(csvPath));
        formData.append('config', JSON.stringify(TEST_CONFIG));

        const response = await axios.post(`${API_BASE_URL}/config`, formData, {
            headers: {
                ...formData.getHeaders(),
            },
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        console.log('✓ Configuration created successfully');
        return response.data.data;
    } catch (error) {
        console.error('Configuration creation failed:', error.response?.data || error.message);
        throw error;
    }
}

async function monitorTraining(configId) {
    console.log(`\nMonitoring training progress for config ${configId}...`);
    const startTime = Date.now();
    let status = { status: 'pending' };

    while (['pending', 'queued', 'processing'].includes(status.status)) {
        if (Date.now() - startTime > TRAINING_TIMEOUT) {
            throw new Error('Training timeout exceeded');
        }

        try {
            const response = await axios.get(`${API_BASE_URL}/config/status/${configId}`);
            status = response.data;

            // Log progress updates
            if (status.training_stats?.progress !== undefined) {
                console.log(`Progress: ${status.training_stats.progress}%`);
            }

            if (status.status === 'failed') {
                throw new Error(`Training failed: ${status.error || 'Unknown error'}`);
            }

            if (status.status === 'completed') {
                console.log('✓ Training completed successfully');
                if (status.training_stats) {
                    console.log('Training Stats:');
                    console.log(`- Total Records: ${status.training_stats.total_records}`);
                    console.log(`- Processed Records: ${status.training_stats.processed_records}`);
                    console.log(`- Training Accuracy: ${status.training_stats.training_accuracy?.toFixed(4)}`);
                    console.log(`- Validation Score: ${status.training_stats.validation_score?.toFixed(4)}`);
                }
                break;
            }

            await new Promise(resolve => setTimeout(resolve, CHECK_INTERVAL));
        } catch (error) {
            console.error('Status check error:', error.message);
            await new Promise(resolve => setTimeout(resolve, CHECK_INTERVAL));
        }
    }

    return status;
}

async function testSearch(query, filters = {}) {
    try {
        console.log(`\nTesting search query: "${query}"`);
        
        const response = await axios.post(`${API_BASE_URL}/search`, {
            query,
            config_id: 'latest',
            max_items: 5,
            filters
        });

        console.log(`Found ${response.data.total} results`);

        response.data.results?.forEach((result, index) => {
            console.log(`\nResult ${index + 1}:`);
            console.log(`- Name: ${result.name}`);
            console.log(`- Category: ${result.category}`);
            console.log(`- Score: ${(result.score * 100).toFixed(1)}%`);
            
            // Log metadata if available
            if (Object.keys(result.metadata || {}).length > 0) {
                console.log('- Metadata:');
                Object.entries(result.metadata).forEach(([key, value]) => {
                    console.log(`  ${key}: ${value}`);
                });
            }
        });

        return response.data;
    } catch (error) {
        console.error('Search error:', error.response?.data || error.message);
        throw error;
    }
}

async function runTestFlow() {
    try {
        console.log('Starting product search test flow...\n');

        // Wait for services to be ready
        await waitForService(API_BASE_URL);

        // Create configuration with CSV file
        const csvPath = path.join(__dirname, 'products.csv');
        const configData = await createConfiguration(csvPath);
        
        console.log('Configuration created:');
        console.log('- Config ID:', configData.config_id);
        console.log('- Model Path:', configData.model_path);

        // Monitor training progress
        const status = await monitorTraining(configData.config_id);
        
        if (status.status === 'completed') {
            // Test various search scenarios
            const testScenarios = [
                {
                    query: "refrigerator with ice maker",
                    filters: {}
                },
                {
                    query: "washing machine",
                    filters: {
                        price: [0, 1000]
                    }
                },
                {
                    query: "dishwasher",
                    filters: {
                        brand: "Bosch"
                    }
                },
                {
                    query: "microwave",
                    filters: {
                        category: "Kitchen Appliances"
                    }
                }
            ];

            for (const scenario of testScenarios) {
                console.log('\nTesting scenario:', JSON.stringify(scenario));
                await testSearch(scenario.query, scenario.filters);
                await new Promise(resolve => setTimeout(resolve, 1000));
            }

            console.log('\n✓ All test scenarios completed successfully!');
        }

    } catch (error) {
        console.error('\nTest flow failed:', error.response?.data || error.message);
        process.exit(1);
    }
}

// Export functions for use in GitHub Actions
module.exports = {
    runTestFlow,
    testSearch,
    waitForService,
    createConfiguration,
    monitorTraining
};

// Run the test if called directly
if (require.main === module) {
    runTestFlow();
}