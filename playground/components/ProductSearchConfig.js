import React, { useState, useCallback, useEffect } from 'react';
import { AlertCircle, Upload, CheckCircle2, HelpCircle, Search, Loader2 } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Form Field Component
const FormField = ({ label, tooltip, error, children }) => (
    <div className="space-y-2">
        <label className="flex items-center space-x-2">
            <span className="text-sm font-medium">{label}</span>
            {tooltip && (
                <div className="relative group">
                    <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    <div className="absolute hidden group-hover:block z-50 w-48 p-2 text-sm bg-black text-white rounded shadow-lg -top-2 left-6">
                        {tooltip}
                    </div>
                </div>
            )}
        </label>
        {children}
        {error && <p className="text-sm text-red-500">{error}</p>}
    </div>
);

// Search Results Component
const SearchResults = ({ results }) => (
    <div className="space-y-3">
        {results.map((result, index) => (
            <div
                key={index}
                className="p-4 border rounded-lg hover:shadow-md transition-shadow duration-200 bg-white"
            >
                <h4 className="font-semibold text-lg">{result.name}</h4>
                <p className="text-sm text-gray-600 mb-2">{result.description}</p>
                <div className="flex justify-between items-center text-sm text-gray-500">
                    <span className="bg-gray-100 px-2 py-1 rounded-full text-xs">
                        {result.category}
                    </span>
                    <span className="font-medium">
                        Match: {(result.score * 100).toFixed(1)}%
                    </span>
                </div>
            </div>
        ))}
    </div>
);

// Navigation Prompt Dialog
const NavigationPrompt = ({ isOpen, onConfirm, onCancel }) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
            <div className="bg-white p-6 rounded-lg max-w-md w-full mx-4">
                <h3 className="text-lg font-semibold mb-2">Are you sure you want to leave?</h3>
                <p className="text-gray-600 mb-4">
                    You have unsaved changes. If you leave, your training progress will be lost.
                </p>
                <div className="flex justify-end space-x-2">
                    <button
                        onClick={onCancel}
                        className="px-4 py-2 border rounded hover:bg-gray-50"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={onConfirm}
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                        Continue
                    </button>
                </div>
            </div>
        </div>
    );
};

// Basic Info Step
const BasicInfoStep = ({ config, setConfig, onNext }) => {
    const [errors, setErrors] = useState({});

    const validate = () => {
        const newErrors = {};
        if (!config.name.trim()) newErrors.name = 'Name is required';
        if (!config.description.trim()) newErrors.description = 'Description is required';
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleNext = () => {
        if (validate()) onNext();
    };

    return (
        <div className="space-y-6 p-6">
            <h2 className="text-xl font-semibold">Basic Information</h2>
            <div className="space-y-4">
                <FormField
                    label="Configuration Name"
                    tooltip="Give your search configuration a descriptive name"
                    error={errors.name}
                >
                    <input
                        type="text"
                        className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        value={config.name}
                        onChange={e => setConfig(prev => ({ ...prev, name: e.target.value }))}
                        placeholder="Enter configuration name"
                    />
                </FormField>
                <FormField
                    label="Description"
                    tooltip="Describe the purpose of this search configuration"
                    error={errors.description}
                >
                    <textarea
                        className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none min-h-[100px]"
                        value={config.description}
                        onChange={e => setConfig(prev => ({ ...prev, description: e.target.value }))}
                        placeholder="Enter description"
                    />
                </FormField>
                <button
                    className="w-full sm:w-auto px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                    onClick={handleNext}
                >
                    Next
                </button>
            </div>
        </div>
    );
};

// CSV Upload Step
const CsvUploadStep = ({ onFileUpload, csvHeaders, csvFile, onBack, onNext, isUploading }) => {
    const [dragActive, setDragActive] = useState(false);

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        const files = e.dataTransfer.files;
        if (files && files[0]) {
            const file = files[0];
            if (file.type === "text/csv" || file.name.endsWith('.csv')) {
                onFileUpload({ target: { files: [file] } });
            }
        }
    }, [onFileUpload]);

    const handleFileInput = useCallback((e) => {
        const file = e.target.files?.[0];
        if (file && (file.type === "text/csv" || file.name.endsWith('.csv'))) {
            onFileUpload(e);
        }
    }, [onFileUpload]);

    return (
        <div className="space-y-6 p-6">
            <h2 className="text-xl font-semibold">Upload Product Data</h2>
            <div 
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${
                    dragActive ? 'border-blue-500 bg-blue-50' : 'hover:border-blue-500'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <div className="flex flex-col items-center space-y-4">
                    <Upload className="h-12 w-12 text-gray-400" />
                    <div className="space-y-2">
                        <h3 className="font-semibold">Upload CSV file</h3>
                        <p className="text-sm text-gray-500">
                            Drag and drop or click to select
                        </p>
                    </div>
                    <label className="relative cursor-pointer">
                        <input
                            type="file"
                            className="hidden"
                            accept=".csv,text/csv"
                            onChange={handleFileInput}
                            disabled={isUploading}
                        />
                        <button
                            type="button"
                            className={`px-4 py-2 border rounded ${
                                isUploading ? 'bg-gray-100' : 'hover:bg-gray-50'
                            }`}
                            disabled={isUploading}
                            onClick={() => document.querySelector('input[type="file"]').click()}
                        >
                            {isUploading ? (
                                <span className="flex items-center">
                                    <Loader2 className="animate-spin mr-2 h-4 w-4" />
                                    Uploading...
                                </span>
                            ) : (
                                'Select File'
                            )}
                        </button>
                    </label>
                </div>
            </div>

            {csvFile && (
                <div className="bg-green-50 border border-green-200 rounded p-4 flex items-center">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mr-2" />
                    <span>Successfully loaded: {csvFile.name}</span>
                </div>
            )}

            {csvHeaders.length > 0 && (
                <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-2">Detected Columns</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                        {csvHeaders.map(header => (
                            <div key={header} className="px-3 py-1 bg-gray-100 rounded-full text-sm text-center">
                                {header}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="flex justify-between pt-4">
                <button 
                    className="px-4 py-2 border rounded hover:bg-gray-50"
                    onClick={onBack}
                >
                    Back
                </button>
                <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                    onClick={onNext}
                    disabled={!csvFile || isUploading}
                >
                    Next
                </button>
            </div>
        </div>
    );
};
// Column Mapping Step
const ColumnMappingStep = ({ config, setConfig, csvHeaders, onBack, onSubmit, loading }) => {
    const [errors, setErrors] = useState({});

    const validate = () => {
        const newErrors = {};
        if (!config.schema_mapping.id_column) newErrors.id = 'ID column is required';
        if (!config.schema_mapping.name_column) newErrors.name = 'Name column is required';
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = () => {
        if (validate()) onSubmit();
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

    return (
        <div className="space-y-6 p-6">
            <h2 className="text-xl font-semibold">Column Mapping</h2>
            <div className="space-y-6">
                <div className="grid gap-4">
                    {['id', 'name', 'description', 'category'].map(field => (
                        <FormField
                            key={field}
                            label={`${field.charAt(0).toUpperCase() + field.slice(1)} Column`}
                            tooltip={`Select the column that contains ${field} information`}
                            error={errors[field]}
                        >
                            <select
                                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
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

                <div className="border rounded-lg p-4 space-y-4">
                    <div className="flex justify-between items-center">
                        <h3 className="font-semibold">Custom Columns</h3>
                        <button
                            className="px-3 py-1 text-sm border rounded hover:bg-gray-50"
                            onClick={handleCustomColumnAdd}
                        >
                            Add Column
                        </button>
                    </div>

                    {config.schema_mapping.custom_columns.map((col, index) => (
                        <div key={index} className="flex gap-2 items-center">
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
                                className="p-2 text-red-500 hover:text-red-600"
                                onClick={() => {
                                    const newColumns = config.schema_mapping.custom_columns.filter((_, i) => i !== index);
                                    setConfig(prev => ({
                                        ...prev,
                                        schema_mapping: {
                                            ...prev.schema_mapping,
                                            custom_columns: newColumns
                                        }
                                    }));
                                }}
                            >
                                ×
                            </button>
                        </div>
                    ))}
                </div>

                <div className="flex justify-between pt-4">
                    <button
                        className="px-4 py-2 border rounded hover:bg-gray-50"
                        onClick={onBack}
                    >
                        Back
                    </button>
                    <button
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 flex items-center"
                        onClick={handleSubmit}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Processing...
                            </>
                        ) : (
                            'Start Training'
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
};

// Search Box Component
const SearchBox = ({
    selectedVersion,
    setSelectedVersion,
    modelVersions,
    searchQuery,
    setSearchQuery,
    handleSearch,
    loading
}) => (
    <div className="border rounded-lg p-6 mt-8">
        <h3 className="font-semibold mb-4">Search Products</h3>
        <div className="space-y-4">
            <select
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
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

            <div className="flex space-x-2">
                <input
                    type="text"
                    className="flex-1 p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                    placeholder="Enter search query..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key === 'Enter' && !loading) {
                            handleSearch();
                        }
                    }}
                />
                <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 flex items-center"
                    onClick={handleSearch}
                    disabled={loading || !searchQuery}
                >
                    {loading ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                        <Search className="h-4 w-4" />
                    )}
                </button>
            </div>
        </div>
    </div>
);

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
    const [showNavigationPrompt, setShowNavigationPrompt] = useState(false);
    const [pendingNavigation, setPendingNavigation] = useState(null);
    const [isUploading, setIsUploading] = useState(false);

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

    const handleNavigationAttempt = (targetStep) => {
        if (loading) {
            setShowNavigationPrompt(true);
            setPendingNavigation(targetStep);
            return false;
        }
        return true;
    };

    const handleNavigationConfirm = () => {
        setShowNavigationPrompt(false);
        if (pendingNavigation !== null) {
            setStep(pendingNavigation);
            setPendingNavigation(null);
        }
    };

    const handleFileUpload = useCallback(async (e) => {
        const file = e.target.files?.[0];
        if (!file || !file.name.toLowerCase().endsWith('.csv')) {
            setError('Please upload a CSV file');
            return;
        }

        setIsUploading(true);
        setError('');
        
        try {
            setCsvFile(file);
            const filename = `${Date.now()}-${file.name}`;
            const filepath = `/data/products/${filename}`;

            setConfig(prev => ({
                ...prev,
                data_source: {
                    ...prev.data_source,
                    location: filepath
                }
            }));

            // Read file content
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const text = event.target.result;
                    const lines = text.split('\n');
                    if (lines.length > 0) {
                        const headers = lines[0].trim().split(',').map(h => h.trim());
                        setCsvHeaders(headers);
                        setConfig(prev => ({
                            ...prev,
                            data_source: {
                                ...prev.data_source,
                                columns: headers.map(header => ({
                                    name: header,
                                    type: 'string',
                                    role: 'data'
                                }))
                            }
                        }));
                    }
                } catch (err) {
                    setError('Failed to parse CSV file');
                    setCsvFile(null);
                    setCsvHeaders([]);
                }
            };

            reader.onerror = () => {
                setError('Failed to read CSV file');
                setCsvFile(null);
                setCsvHeaders([]);
            };

            reader.readAsText(file);
        } catch (err) {
            setError('Failed to process the file');
            setCsvFile(null);
            setCsvHeaders([]);
        } finally {
            setIsUploading(false);
        }
    }, []);

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
                setSuccess('Training started successfully!');
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
                    fetchModelVersions(); // Refresh model versions list
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

            <NavigationPrompt
                isOpen={showNavigationPrompt}
                onConfirm={handleNavigationConfirm}
                onCancel={() => setShowNavigationPrompt(false)}
            />

            {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center text-red-700">
                    <AlertCircle className="h-4 w-4 mr-2" />
                    <p>{error}</p>
                </div>
            )}

            {success && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center text-green-700">
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    <p>{success}</p>
                </div>
            )}

            <SearchBox
                selectedVersion={selectedVersion}
                setSelectedVersion={setSelectedVersion}
                modelVersions={modelVersions}
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                handleSearch={handleSearch}
                loading={loading}
            />

            <div className="border rounded-lg">
                {step === 1 && (
                    <BasicInfoStep
                        config={config}
                        setConfig={setConfig}
                        onNext={() => handleNavigationAttempt(2) && setStep(2)}
                    />
                )}

                {step === 2 && (
                    <CsvUploadStep
                        onFileUpload={handleFileUpload}
                        csvHeaders={csvHeaders}
                        csvFile={csvFile}
                        onBack={() => setStep(1)}
                        onNext={() => setStep(3)}
                        isUploading={isUploading}
                    />
                )}

                {step === 3 && (
                    <ColumnMappingStep
                        config={config}
                        setConfig={setConfig}
                        csvHeaders={csvHeaders}
                        onBack={() => handleNavigationAttempt(2) && setStep(2)}
                        onSubmit={handleSubmit}
                        loading={loading}
                    />
                )}
            </div>

            {trainingStatus && (
                <div className="border rounded-lg p-6 space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="font-semibold">Training Status</h3>
                        <span className="capitalize">{trainingStatus.status}</span>
                    </div>

                    <div className="relative w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                            className="absolute top-0 left-0 h-full bg-blue-500 transition-all duration-300"
                            style={{
                                width: `${trainingStatus.status === 'completed' ? 100 :
                                    trainingStatus.status === 'failed' ? 0 :
                                        trainingStatus.progress || 50}%`
                            }}
                        />
                    </div>

                    {trainingStatus.error && (
                        <p className="text-sm text-red-500">{trainingStatus.error}</p>
                    )}
                </div>
            )}

            {searchResults.length > 0 && <SearchResults results={searchResults} />}
        </div>
    );
}