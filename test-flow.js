// test-flow.js
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const fsSync = require('fs');

const API_BASE_URL = 'http://localhost:8080';
const TEST_CONFIG = {
    name: 'Test Product Search',
    description: 'Test configuration for product search',
    data_source: {
        type: 'csv',
        location: './data/products/test.csv'
    },
    training_config: {
        model_type: 'transformer',
        embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
        zero_shot_model: 'facebook/bart-large-mnli',
        batch_size: 32,
        max_tokens: 512
    },
    schema_mapping: {
        id_column: 'id',
        name_column: 'name',
        description_column: 'description',
        category_column: 'category',
        custom_columns: [
            {
                user_column: 'price',
                standard_column: 'price',
                role: 'metadata'
            },
            {
                user_column: 'tags',
                standard_column: 'tags',
                role: 'training'
            }
        ]
    }
};

const csvFilePath = path.join(__dirname, 'products.csv');
const SAMPLE_CSV = fsSync.readFileSync(csvFilePath, 'utf8');

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function createTestData() {
    // Create test data directory
    const dataDir = path.join(__dirname, 'data', 'products');
    await fs.mkdir(dataDir, { recursive: true });
    
    // Write test CSV
    const csvPath = path.join(dataDir, 'test.csv');
    await fs.writeFile(csvPath, SAMPLE_CSV);
    
    console.log('Test data created successfully');
}

async function testProductSearch() {
    try {
        console.log('Starting test flow...');

        // Create test data
        await createTestData();
        console.log('✓ Test data created');

        // Create config
        const configResponse = await axios.post(`${API_BASE_URL}/config`, TEST_CONFIG);
        const configId = configResponse.data.id;
        console.log('✓ Configuration created:', configId);

        // Upload products
        const uploadResponse = await axios.post(`${API_BASE_URL}/products/update`, {
            config_id: configId,
            mode: 'replace',
            csv_content: SAMPLE_CSV
        });
        console.log('✓ Products uploaded:', uploadResponse.data.count, 'items');

        // Trigger training
        const trainingResponse = await axios.post(`${API_BASE_URL}/model/train`, {
            config_id: configId
        });
        const versionId = trainingResponse.data.id;
        console.log('✓ Training started:', versionId);

        // Poll training status
        let status = 'training';
        while (status === 'training') {
            await sleep(5000); // Check every 5 seconds
            const statusResponse = await axios.get(`${API_BASE_URL}/model/version/${versionId}`);
            status = statusResponse.data.status;
            console.log('Training status:', status);
        }

        if (status === 'failed') {
            throw new Error('Training failed');
        }

        console.log('✓ Training completed');

        // Test different search queries
        const searchQueries = [
            'need something for cooking',
           
        ];

        for (const query of searchQueries) {
            const searchResponse = await axios.post(`${API_BASE_URL}/search`, {
                query: query,
                version: 'latest',
                max_items: 3
            });

            console.log('\nSearch Results for:', query);
            searchResponse.data.results.forEach(result => {
                console.log(`- ${result.name} (Score: ${result.score.toFixed(3)})`);
            });
        }

        console.log('\nTest flow completed successfully!');

    } catch (error) {
        console.error('Error in test flow:', error.response?.data || error.message);
        process.exit(1);
    }
}

testProductSearch();