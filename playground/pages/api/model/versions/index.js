// pages/api/model/versions.ts
export default async function handler(req, res) {
    try {
      const response = await fetch('http://localhost:8080/model/versions');
      const data = await response.json();
      res.status(200).json(data);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch model versions' });
    }
  }