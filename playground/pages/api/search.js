// pages/api/search.js
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8080';

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Validate request body
    const { query, version = 'latest', max_items = 10 } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    if (typeof query !== 'string') {
      return res.status(400).json({ error: 'Query must be a string' });
    }

    if (max_items && (isNaN(max_items) || max_items < 1 || max_items > 100)) {
      return res.status(400).json({ error: 'max_items must be between 1 and 100' });
    }

    // Prepare request to backend service
    const searchRequest = {
      query,
      version,
      max_items: Number(max_items)
    };

    // Call backend search service
    const response = await axios.post(`${BACKEND_URL}/search`, searchRequest, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000 // 30 second timeout
    });

    const data = response.data;

    // Validate response data
    if (!Array.isArray(data.results)) {
      return res.status(500).json({ error: 'Invalid response from search service' });
    }

    // Format and normalize scores
    const results = data.results.map(result => ({
      ...result,
      score: Number(result.score) || 0,
      name: result.name || 'Untitled Product',
      description: result.description || 'No description available'
    }));

    return res.status(200).json({
      results,
      total: data.total || results.length
    });

  } catch (error) {
    // Log error for debugging (you should use proper logging in production)
    console.error('Search error:', error);

    // Determine if error is a timeout
    const isTimeout = error.code === 'ECONNABORTED' || error.message?.includes('timeout');
    const status = isTimeout ? 504 : 500;
    const message = isTimeout ? 'Search request timed out' : 'Internal server error';

    return res.status(status).json({ error: message });
  }
}

// Configure API route to handle larger request bodies if needed
export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
  },
};