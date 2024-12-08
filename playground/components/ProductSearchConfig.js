import React, { useState, useCallback, useEffect } from 'react';
import { AlertCircle, Upload, CheckCircle2, HelpCircle, Search } from 'lucide-react';
import { Alert, AlertDescription } from './ui/alert';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "./ui/tooltip";

export default function ProductSearchConfig() {
    const [step, setStep] = useState(1);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [csvHeaders, setCsvHeaders] = useState([]);
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [modelVersions, setModelVersions] = useState([]);
    const [selectedVersion, setSelectedVersion] = useState('latest');
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [config, setConfig] = useState({
        name: '',
        description: '',
        data_source: {
            type: 'csv',  // Set default type
            location: '',  // This will be set when file is uploaded
            columns: []    // Will be populated from CSV headers
        },
        schema_mapping: {
            id_column: '',
            name_column: '',
            description_column: '',
            category_column: '',
            custom_columns: []
        },
        training_config: {
            model_type: 'transformer',
            embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
            zero_shot_model: 'facebook/bart-large-mnli',
            batch_size: 32,
            max_tokens: 512
        }
    });

    useEffect(() => {
        fetchModelVersions();
    }, []);

    const fetchModelVersions = async () => {
        try {
            const response = await fetch('/api/model/versions');
            if (response.ok) {
                const data = await response.json();
                setModelVersions(data);
            }
        } catch (err) {
            console.error('Failed to fetch model versions:', err);
        }
    };

    const [csvFile, setCsvFile] = useState(null);
    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setCsvFile(file);

        // Generate a unique filename or use the original name
        const filename = `${Date.now()}-${file.name}`;
        const filepath = `/data/products/${filename}`;

        setConfig(prev => ({
            ...prev,
            data_source: {
                ...prev.data_source,
                location: filepath,
                columns: [] // Will be updated after parsing CSV
            }
        }));

        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            const lines = text.split('\n');
            if (lines.length > 0) {
                const headers = lines[0].trim().split(',');
                setCsvHeaders(headers);

                // Update data_source columns
                setConfig(prev => ({
                    ...prev,
                    data_source: {
                        ...prev.data_source,
                        columns: headers.map(header => ({
                            name: header,
                            type: 'string',
                            role: 'data',
                            description: `Column ${header}`
                        }))
                    }
                }));
            }
        };

        reader.readAsText(file);
    };

    const handleCustomColumnAdd = () => {
        setConfig(prev => ({
            ...prev,
            schema_mapping: {
                ...prev.schema_mapping,
                custom_columns: [
                    ...prev.schema_mapping.custom_columns,
                    { user_column: '', standard_column: '', role: 'metadata' }
                ]
            }
        }));
    };

    const handleCustomColumnRemove = (index) => {
        setConfig(prev => ({
            ...prev,
            schema_mapping: {
                ...prev.schema_mapping,
                custom_columns: prev.schema_mapping.custom_columns.filter((_, i) => i !== index)
            }
        }));
    };
    const handleSearch = async () => {
        try {
            setLoading(true);
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: searchQuery,
                    version: selectedVersion,
                    max_items: 5
                })
            });

            if (!response.ok) {
                throw new Error('Search failed');
            }

            const data = await response.json();
            setSearchResults(data.results);
            setError('');
        } catch (err) {
            setError('Failed to perform search: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async () => {
        try {
            setLoading(true);
            setError('');

            // Create configuration
            const configResponse = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (!configResponse.ok) throw new Error('Failed to create configuration');

            const configData = await configResponse.json();
            const configId = configData.id;

            // Upload CSV
            const reader = new FileReader();
            reader.onload = async (e) => {
                const csvContent = e.target.result;

                const uploadResponse = await fetch('/api/products/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        config_id: configId,
                        mode: 'replace',
                        csv_content: csvContent
                    })
                });

                if (!uploadResponse.ok) throw new Error('Failed to upload products');

                // Start training
                const trainingResponse = await fetch('/api/model/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ config_id: configId })
                });

                if (!trainingResponse.ok) throw new Error('Failed to start training');

                const trainingData = await trainingResponse.json();
                startTrainingMonitor(trainingData.id);
            };

            reader.readAsText(csvFile);

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const startTrainingMonitor = useCallback(async (versionId) => {
        setTrainingStatus({ status: 'training', progress: 0 });

        const checkStatus = async () => {
            try {
                const response = await fetch(`/api/model/version/${versionId}`);
                const data = await response.json();

                setTrainingStatus(data);

                if (data.status === 'training') {
                    setTimeout(checkStatus, 5000);
                } else if (data.status === 'completed') {
                    setSuccess('Training completed successfully!');
                } else if (data.status === 'failed') {
                    setError(`Training failed: ${data.error}`);
                }
            } catch (err) {
                setError('Failed to check training status');
            }
        };

        checkStatus();
    }, []);

    return (
        <div className="max-w-4xl mx-auto p-6 space-y-8">
            <h1 className="text-2xl font-bold mb-6">Product Search Configuration</h1>

            {error && (
                <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {success && (
                <Alert className="bg-green-50">
                    <CheckCircle2 className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-600">{success}</AlertDescription>
                </Alert>
            )}
            <div className="mt-8 border rounded p-4">
                <h3 className="font-semibold mb-4">Search Products</h3>
                <div className="space-y-4">
                    <div className="flex space-x-4">
                        <select
                            className="p-2 border rounded flex-1"
                            value={selectedVersion}
                            onChange={(e) => setSelectedVersion(e.target.value)}
                        >
                            <option value="latest">Latest Model</option>
                            {modelVersions.map((version) => (
                                <option key={version.id} value={version.id}>
                                    {version.version} ({version.status})
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="flex space-x-2">
                        <input
                            type="text"
                            className="flex-1 p-2 border rounded"
                            placeholder="Enter search query..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyPress={(e) => {
                                if (e.key === 'Enter') {
                                    handleSearch();
                                }
                            }}
                        />
                        <button
                            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                            onClick={handleSearch}
                            disabled={loading || !searchQuery}
                        >
                            <Search className="h-4 w-4" />
                        </button>
                    </div>

                    {searchResults.length > 0 && (
                        <div className="space-y-2">
                            {searchResults.map((result, index) => (
                                <div key={index} className="p-3 border rounded hover:bg-gray-50">
                                    <h4 className="font-semibold">{result.name}</h4>
                                    <p className="text-sm text-gray-600">{result.description}</p>
                                    <div className="mt-1 flex justify-between text-sm text-gray-500">
                                        <span>Category: {result.category}</span>
                                        <span>Score: {(result.score * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {searchResults.length === 0 && searchQuery && !loading && (
                        <p className="text-gray-500 text-center py-4">
                            No results found for your query.
                        </p>
                    )}
                </div>
            </div>


            <div className="space-y-6">
                {/* Step 1: Basic Info */}
                <div className={`space-y-4 ${step !== 1 && 'hidden'}`}>
                    <h2 className="text-xl font-semibold">Basic Information</h2>

                    <div className="space-y-2">
                        <label className="flex items-center space-x-2">
                            <span>Configuration Name</span>
                            <TooltipProvider>
                                <Tooltip>
                                    <TooltipTrigger>
                                        <HelpCircle className="h-4 w-4" />
                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Give your search configuration a descriptive name</p>
                                    </TooltipContent>
                                </Tooltip>
                            </TooltipProvider>
                        </label>
                        <input
                            type="text"
                            className="w-full p-2 border rounded"
                            value={config.name}
                            onChange={e => setConfig(prev => ({ ...prev, name: e.target.value }))}
                        />
                    </div>

                    <div className="space-y-2">
                        <label className="flex items-center space-x-2">
                            <span>Description</span>
                            <TooltipProvider>
                                <Tooltip>
                                    <TooltipTrigger>
                                        <HelpCircle className="h-4 w-4" />
                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Describe the purpose of this search configuration</p>
                                    </TooltipContent>
                                </Tooltip>
                            </TooltipProvider>
                        </label>
                        <textarea
                            className="w-full p-2 border rounded"
                            value={config.description}
                            onChange={e => setConfig(prev => ({ ...prev, description: e.target.value }))}
                        />
                    </div>

                    <button
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                        onClick={() => setStep(2)}
                    >
                        Next
                    </button>
                </div>

                {/* Step 2: CSV Upload */}
                <div className={`space-y-4 ${step !== 2 && 'hidden'}`}>
                    <h2 className="text-xl font-semibold">Upload Product Data</h2>

                    <div className="border-2 border-dashed rounded-lg p-6 text-center">
                        <div className="flex flex-col items-center space-y-2">
                            <Upload className="h-8 w-8 text-gray-400" />
                            <label className="cursor-pointer text-blue-500 hover:text-blue-600">
                                <span>Upload CSV file</span>
                                <input
                                    type="file"
                                    className="hidden"
                                    accept=".csv"
                                    onChange={handleFileUpload}
                                />
                            </label>
                        </div>
                    </div>

                    {csvHeaders.length > 0 && (
                        <div>
                            <p className="font-semibold">Detected Columns:</p>
                            <ul className="list-disc list-inside">
                                {csvHeaders.map(header => (
                                    <li key={header}>{header}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    <div className="flex space-x-2">
                        <button
                            className="px-4 py-2 border rounded hover:bg-gray-50"
                            onClick={() => setStep(1)}
                        >
                            Back
                        </button>
                        <button
                            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            onClick={() => setStep(3)}
                            disabled={!csvFile}
                        >
                            Next
                        </button>
                    </div>
                </div>

                {/* Step 3: Column Mapping */}
                <div className={`space-y-4 ${step !== 3 && 'hidden'}`}>
                    <h2 className="text-xl font-semibold">Column Mapping</h2>

                    <div className="space-y-4">
                        {['id', 'name', 'description', 'category'].map(field => (
                            <div key={field} className="space-y-2">
                                <label className="flex items-center space-x-2">
                                    <span className="capitalize">{field} Column</span>
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <HelpCircle className="h-4 w-4" />
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>Select the column that contains {field} information</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </label>
                                <select
                                    className="w-full p-2 border rounded"
                                    value={config.schema_mapping[`${field}_column`]}
                                    onChange={e => setConfig(prev => ({
                                        ...prev,
                                        schema_mapping: {
                                            ...prev.schema_mapping,
                                            [`${field}_column`]: e.target.value
                                        }
                                    }))}
                                >
                                    <option value="">Select column</option>
                                    {csvHeaders.map(header => (
                                        <option key={header} value={header}>
                                            {header}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        ))}
                    </div>

                    <div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <h3 className="font-semibold">Custom Columns</h3>
                            <button
                                className="px-2 py-1 text-sm border rounded hover:bg-gray-50"
                                onClick={handleCustomColumnAdd}
                            >
                                Add Custom Column
                            </button>
                        </div>

                        {config.schema_mapping.custom_columns.map((col, index) => (
                            <div key={index} className="flex space-x-2">
                                <select
                                    className="flex-1 p-2 border rounded"
                                    value={col.user_column}
                                    onChange={e => {
                                        const newColumns = [...config.schema_mapping.custom_columns];
                                        newColumns[index].user_column = e.target.value;
                                        setConfig(prev => ({
                                            ...prev,
                                            schema_mapping: {
                                                ...prev.schema_mapping,
                                                custom_columns: newColumns
                                            }
                                        }));
                                    }}
                                >
                                    <option value="">Select column</option>
                                    {csvHeaders.map(header => (
                                        <option key={header} value={header}>
                                            {header}
                                        </option>
                                    ))}
                                </select>

                                <input
                                    type="text"
                                    className="flex-1 p-2 border rounded"
                                    placeholder="Standard column name"
                                    value={col.standard_column}
                                    onChange={e => {
                                        const newColumns = [...config.schema_mapping.custom_columns];
                                        newColumns[index].standard_column = e.target.value;
                                        setConfig(prev => ({
                                            ...prev,
                                            schema_mapping: {
                                                ...prev.schema_mapping,
                                                custom_columns: newColumns
                                            }
                                        }));
                                    }}
                                />

                                <select
                                    className="p-2 border rounded"
                                    value={col.role}
                                    onChange={e => {
                                        const newColumns = [...config.schema_mapping.custom_columns];
                                        newColumns[index].role = e.target.value;
                                        setConfig(prev => ({
                                            ...prev,
                                            schema_mapping: {
                                                ...prev.schema_mapping,
                                                custom_columns: newColumns
                                            }
                                        }));
                                    }}
                                >
                                    <option value="metadata">Metadata</option>
                                    <option value="training">Training</option>
                                </select>

                                <button
                                    className="px-2 py-1 text-red-500 hover:text-red-600"
                                    onClick={() => handleCustomColumnRemove(index)}
                                >
                                    Remove
                                </button>
                            </div>
                        ))}
                    </div>

                    <div className="flex space-x-2">
                        <button
                            className="px-4 py-2 border rounded hover:bg-gray-50"
                            onClick={() => setStep(2)}
                        >
                            Back
                        </button>
                        <button
                            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            onClick={handleSubmit}
                            disabled={loading}
                        >
                            {loading ? 'Processing...' : 'Start Training'}
                        </button>
                    </div>
                </div>

                {/* Training Status */}
                {trainingStatus && (
                    <div className="mt-8 p-4 border rounded">
                        <h3 className="font-semibold">Training Status</h3>
                        <div className="mt-2">
                            <div className="flex items-center space-x-2">
                                <div className="flex-1 bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-blue-500 h-2 rounded-full"
                                        style={{
                                            width: `${trainingStatus.status === 'completed'
                                                ? '100'
                                                : trainingStatus.status === 'failed'
                                                    ? '0'
                                                    : '50'
                                                }%`
                                        }}
                                    />
                                </div>

                                <span className="text-sm capitalize">
                                    {trainingStatus.status}
                                </span>
                            </div>
                            {trainingStatus.error && (
                                <p className="mt-2 text-red-500">{trainingStatus.error}</p>
                            )}
                        </div>
                    </div>
                )}

                {/* Advanced Settings Section */}
                <div className="mt-8">
                    <details className="border rounded p-4">
                        <summary className="font-semibold cursor-pointer">
                            Advanced Training Settings
                        </summary>
                        <div className="mt-4 space-y-4">
                            <div className="space-y-2">
                                <label className="flex items-center space-x-2">
                                    <span>Embedding Model</span>
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <HelpCircle className="h-4 w-4" />
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>The model used to generate embeddings for your products</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </label>
                                <select
                                    className="w-full p-2 border rounded"
                                    value={config.training_config.embedding_model}
                                    onChange={e => setConfig(prev => ({
                                        ...prev,
                                        training_config: {
                                            ...prev.training_config,
                                            embedding_model: e.target.value
                                        }
                                    }))}
                                >
                                    <option value="sentence-transformers/all-MiniLM-L6-v2">MiniLM-L6</option>
                                    {/* <option value="sentence-transformers/all-mpnet-base-v2">MPNet Base</option>
                                    <option value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2">Multilingual MiniLM</option> */}
                                </select>
                            </div>

                            <div className="space-y-2">
                                <label className="flex items-center space-x-2">
                                    <span>Batch Size</span>
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <HelpCircle className="h-4 w-4" />
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>Number of products processed at once. Larger values use more memory but train faster.</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </label>
                                <input
                                    type="number"
                                    className="w-full p-2 border rounded"
                                    value={config.training_config.batch_size}
                                    onChange={e => setConfig(prev => ({
                                        ...prev,
                                        training_config: {
                                            ...prev.training_config,
                                            batch_size: parseInt(e.target.value)
                                        }
                                    }))}
                                    min={1}
                                    max={128}
                                />
                            </div>

                            <div className="space-y-2">
                                <label className="flex items-center space-x-2">
                                    <span>Max Tokens</span>
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <HelpCircle className="h-4 w-4" />
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>Maximum number of tokens per product description. Longer texts will be truncated.</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </label>
                                <input
                                    type="number"
                                    className="w-full p-2 border rounded"
                                    value={config.training_config.max_tokens}
                                    onChange={e => setConfig(prev => ({
                                        ...prev,
                                        training_config: {
                                            ...prev.training_config,
                                            max_tokens: parseInt(e.target.value)
                                        }
                                    }))}
                                    min={128}
                                    max={1024}
                                />
                            </div>
                        </div>
                    </details>
                </div>

                {/* Results Preview */}
                {trainingStatus?.status === 'completed' && (
                    <div className="mt-8 border rounded p-4">
                        <h3 className="font-semibold mb-4">Test Your Search</h3>
                        <div className="space-y-4">
                            <input
                                type="text"
                                className="w-full p-2 border rounded"
                                placeholder="Enter a search query..."
                                onKeyPress={async (e) => {
                                    if (e.key === 'Enter') {
                                        try {
                                            const response = await fetch('/api/search', {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({
                                                    query: e.target.value,
                                                    version: 'latest',
                                                    max_items: 5
                                                })
                                            });

                                            const results = await response.json();
                                            setSearchResults(results.results);
                                        } catch (err) {
                                            setError('Failed to perform search');
                                        }
                                    }
                                }}
                            />

                            {searchResults?.length > 0 && (
                                <div className="space-y-2">
                                    {searchResults.map((result, index) => (
                                        <div key={index} className="p-3 border rounded hover:bg-gray-50">
                                            <h4 className="font-semibold">{result.name}</h4>
                                            <p className="text-sm text-gray-600">{result.description}</p>
                                            <div className="mt-1 text-sm text-gray-500">
                                                Score: {(result.score * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}