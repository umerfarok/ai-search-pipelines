// test-flow.js
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const fsSync = require('fs');

const API_BASE_URL = 'http://localhost:8080';
const TRAINING_TIMEOUT = 30 * 60 * 1000;
const CHECK_INTERVAL = 5000;

// Configuration matching actual appliance data schema
const TEST_CONFIG = {
    name: 'Appliance Product Search',
    description: 'Product search configuration for appliances dataset',
    data_source: {
        type: 'csv',
        location: '' // Will be set after upload
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
                role: 'training'
            },
            {
                user_column: 'SUBCATEGORY',
                standard_column: 'subcategory',
                role: 'training'
            },
            {
                user_column: 'PRICE_CURRENT',
                standard_column: 'price',
                role: 'metadata'
            },
            {
                user_column: 'SELLER',
                standard_column: 'seller',
                role: 'metadata'
            }
        ]
    }
};

async function readApplianceData() {
    try {
        // Try to read the sample data
        return fsSync.readFileSync(path.join(__dirname, 'sample_products.csv'), 'utf8');
    } catch (error) {
        // Fallback to products.csv if sample_products.csv doesn't exist
        return fsSync.readFileSync(path.join(__dirname, 'products.csv'), 'utf8');
    }
}

async function setupTestData() {
    try {
        console.log('Reading appliance data...');
        const applianceData = await readApplianceData();
        console.log('✓ Data file read successfully');

        console.log('Uploading appliance data...');
        const uploadResponse = await axios.post(`${API_BASE_URL}/data/upload`, {
            file_name: 'appliances.csv',
            content: applianceData
        });

        if (!uploadResponse.data.location) {
            throw new Error('No file location returned from upload');
        }

        console.log('✓ Data uploaded:', uploadResponse.data);
        
        // Update config with uploaded file location
        TEST_CONFIG.data_source.location = uploadResponse.data.location;

        // Create configuration
        console.log('Creating search configuration...');
        const configResponse = await axios.post(`${API_BASE_URL}/config`, TEST_CONFIG);
        
        return {
            configId: configResponse.data.id,
            filePath: uploadResponse.data.location
        };
    } catch (error) {
        console.error('Setup failed:', error.response?.data || error.message);
        throw error;
    }
}

async function monitorTraining(versionId) {
    console.log(`\nMonitoring training progress for version ${versionId}...`);
    const startTime = Date.now();
    let status = 'processing';
    let lastProgress = -1;

    while (['queued', 'processing'].includes(status)) {
        if (Date.now() - startTime > TRAINING_TIMEOUT) {
            throw new Error('Training timeout exceeded');
        }

        try {
            const response = await axios.get(`${API_BASE_URL}/model/version/${versionId}`);
            status = response.data.status;
            
            // Log progress changes
            if (response.data.progress !== lastProgress) {
                lastProgress = response.data.progress;
                if (lastProgress) {
                    console.log(`Progress: ${lastProgress}%`);
                }
            }

            if (status === 'failed') {
                throw new Error(`Training failed: ${response.data.error || 'Unknown error'}`);
            }

            if (status === 'completed') {
                console.log('✓ Training completed successfully');
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

async function testSearch(query) {
    try {
        console.log(`\nTesting search query: "${query}"`);
        
        const response = await axios.post(`${API_BASE_URL}/search`, {
            query,
            version: 'latest',
            max_items: 3
        });

        console.log('Response:', response.data.response);
        console.log(`Found ${response.data.total} results`);

        response.data.results?.forEach((result, index) => {
            console.log(`\nResult ${index + 1}:`);
            console.log(`- Name: ${result.name}`);
            console.log(`- Category: ${result.category}`);
            console.log(`- Brand: ${result.metadata?.brand || 'N/A'}`);
            console.log(`- Score: ${result.score.toFixed(3)}`);
            if (result.metadata?.price) {
                console.log(`- Price: $${result.metadata.price}`);
            }
        });

        return response.data;
    } catch (error) {
        console.error('Search error:', error.response?.data || error.message);
    }
}

async function runTestFlow() {
    try {
        console.log('Starting appliance search test flow...\n');

        // Setup test data and configuration
        const { configId, filePath } = await setupTestData();
        console.log('✓ Setup complete');
        console.log('Config ID:', configId);
        console.log('Data location:', filePath);

        // Start training
        console.log('\nTriggering model training...');
        const trainingResponse = await axios.post(`${API_BASE_URL}/model/train`, {
            config_id: configId
        });
        const versionId = trainingResponse.data.id;
        console.log('✓ Training job created:', versionId);

        // Wait for training completion
        const status = await monitorTraining(versionId);
        
        if (status === 'completed') {
            // Test various search queries
            const testQueries = [
                "refrigerator with ice maker",
                "energy efficient washing machine",
                "stainless steel appliances",
                "kitchen appliance under $500",
                "large capacity freezer",
                "smart home appliances",
                "dishwasher with quiet operation"
            ];

            for (const query of testQueries) {
                await testSearch(query);
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

        console.log('\n✓ Test flow completed successfully!');
    } catch (error) {
        console.error('\nTest flow failed:', error.response?.data || error.message);
        process.exit(1);
    }
}

// Run the test
runTestFlow();