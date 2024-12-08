
// pages/api/model/version/[versionId].js
export default async function handler(req, res) {
    if (req.method !== 'GET') {
      return res.status(405).json({ message: 'Method not allowed' });
    }
  
    const { versionId } = req.query;
  
    if (!versionId) {
      return res.status(400).json({ error: 'Version ID is required' });
    }
  
    try {
      const response = await fetch(`http://localhost:8080/model/version/${versionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
  
      const data = await response.json();
  
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch model version');
      }
  
      return res.status(response.status).json(data);
    } catch (error) {
      console.error('Version fetch error:', error);
      return res.status(500).json({ 
        error: error.message || 'Internal server error' 
      });
    }
  }
  