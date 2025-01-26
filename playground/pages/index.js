import { 
  Database, 
  Search, 
  Brain, 
  Cpu, 
  Box, 
  BarChart, 
  GitBranch,
  Rocket,
  Terminal,
  Clock,
  Plus,
  BookOpen,
  Mail
} from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  const features = [
    {
      icon: Brain,
      title: 'Neural Search',
      description: 'Uses transformer-based deep learning models for semantic understanding'
    },
    {
      icon: Search,
      title: 'Semantic Matching',
      description: 'Understands similar products even with different descriptions'
    },
    {
      icon: Database,
      title: 'Smart Embeddings',
      description: 'Generates vector embeddings using state-of-the-art language models'
    },
    {
      icon: BarChart,
      title: 'Intelligent Ranking',
      description: 'AI-powered relevance scoring and result ranking'
    }
  ];

  const techStack = [
    {
      icon: Box,
      title: 'API Service (Go)',
      features: [
        'Configuration management',
        'Model versioning',
        'Training coordination',
        'Search routing'
      ]
    },
    {
      icon: Cpu,
      title: 'Training Service (Python)',
      features: [
        'Queue-based processing',
        'Embedding generation',
        'Model artifacts storage',
        'Progress tracking'
      ]
    },
    {
      icon: GitBranch,
      title: 'Search Service (Python)',
      features: [
        'Model loading',
        'Semantic search',
        'Result filtering',
        'Response ranking'
      ]
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-gray-900 to-gray-800 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">
              AI-Powered Product Search Engine
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto">
              Advanced semantic search solution using neural networks and transformer models
            </p>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900">AI Features</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="p-6 border rounded-lg hover:shadow-lg transition-shadow">
                <feature.icon className="h-8 w-8 text-blue-500 mb-4" />
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Technical Architecture */}
      <div className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900">Technical Architecture</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {techStack.map((tech, index) => (
              <div key={index} className="p-6 bg-white rounded-lg shadow-md">
                <tech.icon className="h-8 w-8 text-blue-500 mb-4" />
                <h3 className="text-xl font-semibold mb-4">{tech.title}</h3>
                <ul className="space-y-2">
                  {tech.features.map((feature, i) => (
                    <li key={i} className="flex items-center text-gray-600">
                      <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Getting Started */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900">Getting Started</h2>
            <p className="mt-4 text-lg text-gray-600">Follow these steps to start using the AI Search platform</p>
          </div>
          <div className="max-w-4xl mx-auto">
            <div className="space-y-6">
              {/* Step 1 */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center bg-blue-100 rounded-full text-blue-600 font-bold">
                    1
                  </div>
                  <h3 className="ml-4 text-lg font-medium">Create New Configuration</h3>
                </div>
                <p className="text-gray-600 mb-4">
                  Navigate to the "Create Config" section to start setting up your search model.
                </p>
                <Link href="/create-config"
                  className="inline-flex items-center text-blue-600 hover:text-blue-700">
                  <Plus className="h-4 w-4 mr-2" />
                  Create New Config
                </Link>
              </div>

              {/* Step 2 */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center bg-blue-100 rounded-full text-blue-600 font-bold">
                    2
                  </div>
                  <h3 className="ml-4 text-lg font-medium">Upload Your Data</h3>
                </div>
                <div className="space-y-2 text-gray-600">
                  <p>Upload your product data CSV file and configure the schema mapping:</p>
                  <ul className="list-disc list-inside ml-4 space-y-1">
                    <li>Select required columns (ID, Name, Description)</li>
                    <li>Map custom fields for additional filtering</li>
                    <li>Configure training parameters if needed</li>
                  </ul>
                </div>
              </div>

              {/* Step 3 */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center bg-blue-100 rounded-full text-blue-600 font-bold">
                    3
                  </div>
                  <h3 className="ml-4 text-lg font-medium">Monitor Training</h3>
                </div>
                <p className="text-gray-600 mb-4">
                  Wait for the model to complete training. You can monitor the progress in real-time.
                </p>
                <Link href="/pending-jobs"
                  className="inline-flex items-center text-blue-600 hover:text-blue-700">
                  <Clock className="h-4 w-4 mr-2" />
                  View Training Status
                </Link>
              </div>

              {/* Step 4 */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center bg-blue-100 rounded-full text-blue-600 font-bold">
                    4
                  </div>
                  <h3 className="ml-4 text-lg font-medium">Start Searching</h3>
                </div>
                <p className="text-gray-600 mb-4">
                  Once training is complete, go to the Search section to start using your model.
                </p>
                <Link href="/search"
                  className="inline-flex items-center text-blue-600 hover:text-blue-700">
                  <Search className="h-4 w-4 mr-2" />
                  Go to Search
                </Link>
              </div>
            </div>

            {/* Help Section */}
            <div className="mt-8 pt-8 border-t border-gray-200 text-center">
              <p className="text-gray-600">
                Need help getting started?
              </p>
              <div className="mt-4 flex justify-center space-x-4">
                <Link href="/docs"
                  className="text-blue-600 hover:text-blue-700 flex items-center">
                  <BookOpen className="h-4 w-4 mr-2" />
                  Documentation
                </Link>
                <a href="mailto:umerfarooq.dev@gmail.com"
                  className="text-blue-600 hover:text-blue-700 flex items-center">
                  <Mail className="h-4 w-4 mr-2" />
                  Contact Support
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}