// testQueryonly.js
import axios from 'axios';

// const searchQueries = [
//     'bottom freezer refrigerator',
//     'energy star appliance',
//     'stainless steel fridge',
//     'large capacity refrigerator',
//     'counter depth refrigerator',
//     'LG appliance',
//     'refrigerator with ice maker',
//     'affordable refrigerator under 1000',
//     "need something for cooking",
//     "need something for cleaning",
// ];67559e7db6e9095082d468eb
const searchQueries = [
    "hello",
    "need something for cleaning",
];

const API_BASE_URL = 'http://localhost:8080';
async function testSearch(query) {
    console.log('\nTesting query:', query);
    try {
        const response = await axios.post(`${API_BASE_URL}/search`, {
            query,
            version: 'latest',
            max_items: 3
        });

        console.log('Results found:', response.data);
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
async function runSearchQueries() {
    // 
    // models/20241208132621
    try {
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
             // Rate limiting
        }

    } catch (error) {
        console.error('Fatal error:', error.message);
    }
}

runSearchQueries();