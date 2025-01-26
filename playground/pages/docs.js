import { ArrowLeft, Code, GitBranch, Terminal, Settings, Database, Search } from 'lucide-react';
import Link from 'next/link';

const DocSection = ({ title, children }) => (
  <div className="mb-12">
    <h2 className="text-2xl font-bold mb-4 text-gray-900">{title}</h2>
    {children}
  </div>
);

const CodeBlock = ({ code, language = "bash" }) => (
  <div className="bg-gray-900 rounded-lg p-4 my-4">
    <pre className="text-gray-100 font-mono text-sm overflow-x-auto">
      <code>{code}</code>
    </pre>
  </div>
);

export default function Documentation() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Back Button */}
        <Link href="/" className="inline-flex items-center text-gray-600 hover:text-gray-900 mb-8">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Home
        </Link>

        <h1 className="text-4xl font-bold text-gray-900 mb-8">Documentation</h1>

        {/* Table of Contents */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h3 className="text-lg font-semibold mb-4">Contents</h3>
          <ul className="space-y-2">
            <li><a href="#overview" className="text-blue-600 hover:text-blue-800">Overview</a></li>
            <li><a href="#features" className="text-blue-600 hover:text-blue-800">Features</a></li>
            <li><a href="#architecture" className="text-blue-600 hover:text-blue-800">Architecture</a></li>
            <li><a href="#setup" className="text-blue-600 hover:text-blue-800">Setup & Configuration</a></li>
            <li><a href="#api" className="text-blue-600 hover:text-blue-800">API Reference</a></li>
          </ul>
        </div>

        {/* Content */}
        <div className="bg-white rounded-lg shadow-sm p-8">
          <DocSection title="Overview" id="overview">
            <p className="text-gray-600">
              An advanced AI-based search solution that uses neural networks and transformer models 
              to understand and match product descriptions semantically. Built with modern AI/ML 
              technologies, Go, Python, and React.
            </p>
          </DocSection>

          <DocSection title="Features" id="features">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <Search className="h-5 w-5 text-blue-500" />
                  Search Capabilities
                </h3>
                <ul className="list-disc list-inside text-gray-600 space-y-2">
                  <li>Neural Search with transformer models</li>
                  <li>Contextual Understanding</li>
                  <li>Smart Embeddings</li>
                  <li>Intelligent Ranking</li>
                </ul>
              </div>
              <div className="space-y-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <GitBranch className="h-5 w-5 text-blue-500" />
                  Technical Features
                </h3>
                <ul className="list-disc list-inside text-gray-600 space-y-2">
                  <li>Model Management</li>
                  <li>Real-time Training</li>
                  <li>Custom Schema Mapping</li>
                  <li>Advanced Filtering</li>
                </ul>
              </div>
            </div>
          </DocSection>

          <DocSection title="Setup & Configuration" id="setup">
            <div className="space-y-6">
              <div>
                <h3 className="font-semibold mb-2 flex items-center gap-2">
                  <Terminal className="h-5 w-5 text-blue-500" />
                  Environment Variables
                </h3>
                <CodeBlock code={`# API Service
MONGO_URI=mongodb://root:example@localhost:27017
REDIS_HOST=localhost
REDIS_PORT=6379
AWS_ACCESS_KEY=your_access_key
AWS_SECRET_KEY=your_secret_key
S3_BUCKET=your_bucket
AWS_REGION=us-east-1

# Training Service
SERVICE_PORT=5001
MODEL_NAME=all-mpnet-base-v2
BATCH_SIZE=128

# Search Service
SERVICE_PORT=5000
MODEL_CACHE_SIZE=2
MIN_SCORE=0.2`} />
              </div>

              <div>
                <h3 className="font-semibold mb-2 flex items-center gap-2">
                  <Settings className="h-5 w-5 text-blue-500" />
                  Schema Configuration
                </h3>
                <CodeBlock 
                  language="json"
                  code={`{
  "id_column": "product_id",
  "name_column": "title",
  "description_column": "description",
  "category_column": "category",
  "custom_columns": [
    {
      "user_column": "price",
      "standard_column": "price",
      "role": "filter"
    }
  ]
}`} />
              </div>
            </div>
          </DocSection>

          <DocSection title="API Reference" id="api">
            <div className="space-y-6">
              <div>
                <h3 className="font-semibold mb-2">Configuration Management</h3>
                <ul className="space-y-4">
                  <li className="p-4 bg-gray-50 rounded-lg">
                    <code className="text-sm font-mono text-blue-600">POST /config</code>
                    <p className="mt-2 text-gray-600">Create new model configuration</p>
                  </li>
                  <li className="p-4 bg-gray-50 rounded-lg">
                    <code className="text-sm font-mono text-blue-600">GET /config/:id</code>
                    <p className="mt-2 text-gray-600">Get configuration details</p>
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold mb-2">Search Endpoints</h3>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <code className="text-sm font-mono text-blue-600">POST /search</code>
                  <p className="mt-2 text-gray-600">Perform semantic search</p>
                  <CodeBlock 
                    language="json"
                    code={`{
  "query": "search term",
  "config_id": "config_id",
  "max_items": 10,
  "filters": {
    "category": "electronics",
    "price_range": [0, 1000]
  }
}`} />
                </div>
              </div>
            </div>
          </DocSection>
        </div>
      </div>
    </div>
  );
}