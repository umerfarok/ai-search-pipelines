// pages/api/model/train.js
export default async function handler(req, res) {
    if (req.method !== 'POST') {
      return res.status(405).json({ message: 'Method not allowed' });
    }
  
    try {
      const response = await fetch('http://localhost:8080/model/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(req.body),
      });
  
      const data = await response.json();
  
      if (!response.ok) {
        throw new Error(data.error || 'Failed to start training');
      }
  
      return res.status(response.status).json(data);
    } catch (error) {
      console.error('Training error:', error);
      return res.status(500).json({ 
        error: error.message || 'Internal server error' 
      });
    }
  }