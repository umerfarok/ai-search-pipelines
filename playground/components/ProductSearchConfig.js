import React, { useState, useCallback, useEffect } from 'react';
import { AlertCircle, Upload, CheckCircle2, HelpCircle, Search } from 'lucide-react';
import { Alert, AlertDescription } from './ui/alert';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "./ui/tooltip";
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Reusable Components
const FormField = ({ label, tooltip, children }) => (
    <div className="space-y-2">
        <label className="flex items-center space-x-2">
            <span>{label}</span>
            {tooltip && (
                <TooltipProvider>
                    <Tooltip>
                        <TooltipTrigger>
                            <HelpCircle className="h-4 w-4" />
                        </TooltipTrigger>
                        <TooltipContent>
                            <p>{tooltip}</p>
                        </TooltipContent>
                    </Tooltip>
                </TooltipProvider>
            )}
        </label>
        {children}
    </div>
);

const SearchResults = ({ results }) => (
    <div className="space-y-2">
        {results.map((result, index) => (
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
);

// Step Components
const BasicInfoStep = ({ config, setConfig, onNext }) => (
    <div className="space-y-4">
        <h2 className="text-xl font-semibold">Basic Information</h2>
        <FormField label="Configuration Name" tooltip="Give your search configuration a descriptive name">
            <input
                type="text"
                className="w-full p-2 border rounded"
                value={config.name}
                onChange={e => setConfig(prev => ({ ...prev, name: e.target.value }))}
            />
        </FormField>
        <FormField label="Description" tooltip="Describe the purpose of this search configuration">
            <textarea
                className="w-full p-2 border rounded"
                value={config.description}
                onChange={e => setConfig(prev => ({ ...prev, description: e.target.value }))}
            />
        </FormField>
        <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={onNext}
        >
            Next
        </button>
    </div>
);

const CsvUploadStep = ({ onFileUpload, csvHeaders, csvFile, onBack, onNext }) => (
    <div className="space-y-4">
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
                        onChange={onFileUpload}
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
            <button className="px-4 py-2 border rounded hover:bg-gray-50" onClick={onBack}>
                Back
            </button>
            <button
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                onClick={onNext}
                disabled={!csvFile}
            >
                Next
            </button>
        </div>
    </div>
);

const ColumnMappingStep = ({ config, setConfig, csvHeaders, onBack, onSubmit, loading }) => {
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

    return (
        <div className="space-y-4">
            <h2 className="text-xl font-semibold">Column Mapping</h2>
            <div className="space-y-4">
                {['id', 'name', 'description', 'category'].map(field => (
                    <FormField
                        key={field}
                        label={`${field.charAt(0).toUpperCase() + field.slice(1)} Column`}
                        tooltip={`Select the column that contains ${field} information`}
                    >
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
                                <option key={header} value={header}>{header}</option>
                            ))}
                        </select>
                    </FormField>
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
                                <option key={header} value={header}>{header}</option>
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
                <button className="px-4 py-2 border rounded hover:bg-gray-50" onClick={onBack}>
                    Back
                </button>
                <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    onClick={onSubmit}
                    disabled={loading}
                >
                    {loading ? 'Processing...' : 'Start Training'}
                </button>
            </div>
        </div>
    );
};

// Main Component
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
    const [csvFile, setCsvFile] = useState(null);

    const [config, setConfig] = useState({
        name: '',
        description: '',
        data_source: {
            type: 'csv',
            location: '',
            columns: []
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
            batch_size: 128,
            max_tokens: 512
        }
    });

    useEffect(() => {
        fetchModelVersions();
    }, []);

    const fetchModelVersions = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/model/versions`);
            if (response.ok) {
                const data = await response.json();
                setModelVersions(data);
            }
        } catch (err) {
            console.error('Failed to fetch model versions:', err);
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setCsvFile(file);
        const filename = `${Date.now()}-${file.name}`;
        const filepath = `/data/products/${filename}`;

        setConfig(prev => ({
            ...prev,
            data_source: {
                ...prev.data_source,
                location: filepath,
                columns: []
            }
        }));

        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            const lines = text.split('\n');
            if (lines.length > 0) {
                const headers = lines[0].trim().split(',');
                setCsvHeaders(headers);
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

    const handleSearch = async () => {
        try {
            setLoading(true);
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: searchQuery,
                    version: selectedVersion,
                    max_items: 5
                })
            });
    
            if (!response.ok) throw new Error('Search failed');
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
    
            const configResponse = await fetch(`${API_BASE_URL}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
    
            if (!configResponse.ok) throw new Error('Failed to create configuration');
    
            const configData = await configResponse.json();
            const configId = configData.id;
    
            const reader = new FileReader();
            reader.onload = async (e) => {
                const csvContent = e.target.result;
    
                const uploadResponse = await fetch(`${API_BASE_URL}/products/update`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        config_id: configId,
                        mode: 'replace',
                        csv_content: csvContent
                    })
                });
    
                if (!uploadResponse.ok) throw new Error('Failed to upload products');
    
                const trainingResponse = await fetch(`${API_BASE_URL}/model/train`, {
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
                const response = await fetch(`${API_BASE_URL}/model/version/${versionId}`);
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
    

    const SearchBox = () => (
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
            </div>
        </div>
    );

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

            <SearchBox />

            <div className="space-y-6">
                {step === 1 && (
                    <BasicInfoStep
                        config={config}
                        setConfig={setConfig}
                        onNext={() => setStep(2)}
                    />
                )}

                {step === 2 && (
                    <CsvUploadStep
                        onFileUpload={handleFileUpload}
                        csvHeaders={csvHeaders}
                        csvFile={csvFile}
                        onBack={() => setStep(1)}
                        onNext={() => setStep(3)}
                    />
                )}

                {step === 3 && (
                    <ColumnMappingStep
                        config={config}
                        setConfig={setConfig}
                        csvHeaders={csvHeaders}
                        onBack={() => setStep(2)}
                        onSubmit={handleSubmit}
                        loading={loading}
                    />
                )}

                {trainingStatus && (
                    <div className="mt-8 p-4 border rounded">
                        <h3 className="font-semibold">Training Status</h3>
                        <div className="mt-2">
                            <div className="flex items-center space-x-2">
                                <div className="flex-1 bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-blue-500 h-2 rounded-full"
                                        style={{
                                            width: `${trainingStatus.status === 'completed' ? '100' : 
                                                   trainingStatus.status === 'failed' ? '0' : '50'}%`
                                        }}
                                    />
                                </div>
                                <span className="text-sm capitalize">{trainingStatus.status}</span>
                            </div>
                            {trainingStatus.error && (
                                <p className="mt-2 text-red-500">{trainingStatus.error}</p>
                            )}
                        </div>
                    </div>
                )}

                {searchResults.length > 0 && <SearchResults results={searchResults} />}
            </div>
        </div>
    );
}