
// pages/api/products/update.js
export default async function handler(req, res) {
    if (req.method !== 'POST') {
      return res.status(405).json({ message: 'Method not allowed' });
    }
  
    try {
      const { config_id, mode, csv_content } = req.body;
  
      // Validate required fields
      if (!config_id) {
        return res.status(400).json({ error: 'config_id is required' });
      }
      if (!mode || !['append', 'replace'].includes(mode)) {
        return res.status(400).json({ error: 'mode must be either "append" or "replace"' });
      }
      if (!csv_content) {
        return res.status(400).json({ error: 'csv_content is required' });
      }
  
      const response = await fetch('http://localhost:8080/products/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          config_id,
          mode,
          csv_content,
        }),
      });
  
      const data = await response.json();
  
      if (!response.ok) {
        throw new Error(data.error || 'Failed to update products');
      }
  
      return res.status(response.status).json(data);
    } catch (error) {
      console.error('Product update error:', error);
      return res.status(500).json({ 
        error: error.message || 'Internal server error' 
      });
    }
  }