// pages/api/config.js
export default async function handler(req, res) {
    if (req.method !== 'POST') {
      return res.status(405).json({ message: 'Method not allowed' });
    }
  
    try {
      const response = await fetch('http://localhost:8080/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body)
      });
  
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ message: 'Internal server error' });
    }
  }