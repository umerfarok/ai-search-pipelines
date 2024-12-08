const fs = require('fs');
const path = require('path');

const directories = [
  'api/handlers',
  'trainer',
  'models',
  'data'
];

const files = {
  'api/handlers/config.go': '',
  'api/handlers/model.go': '',
  'api/handlers/product.go': '',
  'api/handlers/search.go': '',
  'api/Dockerfile': '',
  'api/go.mod': '',
  'api/main.go': '',
  'trainer/search_service.py': '',
  'trainer/training_service.py': '',
  'trainer/Dockerfile.python': '',
  'trainer/requirements.txt': '',
  'docker-compose.yaml': ''
};

directories.forEach(dir => {
  fs.mkdirSync(dir, { recursive: true });
});

Object.keys(files).forEach(file => {
  fs.writeFileSync(file, files[file]);
});

console.log('Directory structure and files created successfully.');