// test-flow.js
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const fsSync = require('fs');

const API_BASE_URL = 'http://localhost:8080';
const TEST_CONFIG = {
    name: 'Appliance Product Search',
    description: 'Product search configuration for appliances dataset',
    data_source: {
        type: 'csv',
        location: './data/products/appliances.csv'  // Relative path
    },
    training_config: {
        model_type: 'transformer',
        embedding_model: 'all-mpnet-base-v2',
        zero_shot_model: 'facebook/bart-large-mnli',
        batch_size: 128,
        max_tokens: 512
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

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function createTestData() {
    try {
        // Ensure the data directory exists
        const dataDir = path.join(process.cwd(), 'data', 'products');
        await fs.mkdir(dataDir, { recursive: true });
        
        // Copy the test CSV file
        const csvContent = fsSync.readFileSync(path.join(process.cwd(), 'products.csv'), 'utf8');
        const csvPath = path.join(dataDir, 'appliances.csv');
        await fs.writeFile(csvPath, csvContent);
        
        console.log('✓ Test data created at:', csvPath);
    } catch (error) {
        console.error('Error creating test data:', error);
        throw error;
    }
}
const TRAINING_TIMEOUT = 30 * 60 * 1000; // 30 minutes
const CHECK_INTERVAL = 10000; // 10 seconds

async function waitForTraining(versionId) {
    const startTime = Date.now();
    let status = 'training';
    let lastProgress = -1;

    while (status === 'training') {
        if (Date.now() - startTime > TRAINING_TIMEOUT) {
            throw new Error('Training timeout exceeded');
        }

        try {
            const response = await axios.get(`${API_BASE_URL}/model/version/${versionId}`);
            status = response.data.status;
            
            // Check progress
            if (response.data.progress && response.data.progress.progress !== lastProgress) {
                lastProgress = response.data.progress.progress;
                const total = response.data.progress.total;
                const percent = total > 0 ? ((lastProgress / total) * 100).toFixed(1) : 0;
                console.log(`Training progress: ${lastProgress}/${total} (${percent}%)`);
            }

            if (status === 'failed') {
                throw new Error(`Training failed: ${response.data.error || 'Unknown error'}`);
            }

            if (status === 'completed') {
                console.log('Training completed successfully');
                break;
            }

            await sleep(CHECK_INTERVAL);
        } catch (error) {
            console.error('Error checking training status:', error.message);
            await sleep(CHECK_INTERVAL);
        }
    }

    return status;
}
async function testSearch(query) {
    console.log('\nTesting query:', query);
    try {
        const response = await axios.post(`${API_BASE_URL}/search`, {
            query,
            version: 'latest',
            max_items: 3
        });

        console.log('Results found:', response.data.total);
        if (response.data.response) {
            console.log('Response:', response.data.response);
        }

        response.data.results?.forEach((result, index) => {
            console.log(`\nResult ${index + 1}:`);
            console.log(`- Product: ${result.name}`);
            console.log(`- Category: ${result.category}`);
            console.log(`- Score: ${result.score.toFixed(3)}`);
            if (result.metadata) {
                console.log('- Metadata:', JSON.stringify(result.metadata, null, 2));
            }
        });

        return response.data;
    } catch (error) {
        console.error('Search error:', error.response?.data || error.message);
        throw error;
    }
}

async function testApplianceSearch() {
    try {
        console.log('Starting appliance search test flow...');

        // Create and verify test data
        await createTestData();
        console.log('✓ Test data created');

        // Create configuration
        const configResponse = await axios.post(`${API_BASE_URL}/config`, TEST_CONFIG);
        const configId = configResponse.data.id;
        console.log('✓ Configuration created:', configId);

        // Upload products
        const csvContent = fsSync.readFileSync(path.join(process.cwd(), 'products.csv'), 'utf8');
        const uploadResponse = await axios.post(`${API_BASE_URL}/products/update`, {
            config_id: configId,
            mode: 'replace',
            csv_content: csvContent
        });
        console.log('✓ Products uploaded:', uploadResponse.data.count, 'items');

        // Start training
        const trainingResponse = await axios.post(`${API_BASE_URL}/model/train`, {
            config_id: configId
        });
        const versionId = trainingResponse.data.id;
        console.log('✓ Training started:', versionId);

        // Wait for training completion
        const finalStatus = await waitForTraining(versionId);
        console.log('✓ Training completed with status:', finalStatus);

        // Test search queries
        const testQueries = [
            "bottom freezer refrigerator",
            "energy efficient appliance",
            "large capacity fridge",
            "stainless steel refrigerator",
            "I need a new refrigerator",
            "hello",
            "looking for a washing machine",
            "need something for food storage"
        ];

        console.log('\nExecuting test queries...');
        for (const query of testQueries) {
            await testSearch(query);
            await sleep(1000); // Rate limiting
        }

        console.log('\n✓ Test flow completed successfully!');

    } catch (error) {
        console.error('Error in test flow:', error.response?.data || error.message);
        process.exit(1);
    }
}

testApplianceSearch();