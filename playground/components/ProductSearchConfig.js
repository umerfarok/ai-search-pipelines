import React, { useState, useCallback, useEffect } from "react";
import {
    AlertCircle,
    Upload,
    CheckCircle2,
    HelpCircle,
    Search,
    Loader2,
    XCircle,
    Clock,
    AlertTriangle,
} from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

const ModelStatus = {
    PENDING: "pending",
    QUEUED: "queued",
    PROCESSING: "processing",
    COMPLETED: "completed",
    FAILED: "failed",
    CANCELED: "canceled",
};

const useModelConfig = () => {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState("");
    const [configId, setConfigId] = useState(null);
    const [config, setConfig] = useState({
        name: "",
        description: "",
        mode: "replace", // "replace" or "append"
        previous_version: "", // Used for append mode
        data_source: {
            type: "csv",
            location: "",
            file_type: "csv",
            columns: [],
        },
        schema_mapping: {
            id_column: "",
            name_column: "",
            description_column: "",
            category_column: "",
            custom_columns: [],
            required_columns: [],
        },
        training_config: {
            model_type: "transformer",
            embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
            batch_size: 128,
            max_tokens: 512,
            validation_split: 0.2,
            training_params: {},
        },
    });

    const submitConfig = async (file) => {
        setIsSubmitting(true);
        setError("");

        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("config", JSON.stringify(config));

            const response = await fetch(`${API_BASE_URL}/config`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Failed to create configuration");
            }

            const data = await response.json();
            setConfigId(data.data.config_id);
            return data;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setIsSubmitting(false);
        }
    };

    return {
        config,
        setConfig,
        submitConfig,
        isSubmitting,
        error,
        configId,
    };
};

const useFileUpload = () => {
    const [isUploading, setIsUploading] = useState(false);
    const [csvHeaders, setCsvHeaders] = useState([]);
    const [error, setError] = useState("");

    const handleFileUpload = useCallback(async (file, setConfig) => {
        if (!file) {
            setError("No file selected");
            return;
        }

        if (!file.name.toLowerCase().endsWith(".csv")) {
            setError("Please upload a CSV file");
            return;
        }

        setIsUploading(true);
        setError("");

        try {
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const text = event.target.result;
                    const lines = text.split("\n");
                    if (lines.length === 0) {
                        throw new Error("The CSV file is empty");
                    }
                    const headers = lines[0]
                        .trim()
                        .split(",")
                        .map((h) => h.trim());
                    if (headers.length === 0) {
                        throw new Error("No headers found in the CSV file");
                    }
                    setCsvHeaders(headers);
                    setConfig((prev) => ({
                        ...prev,
                        data_source: {
                            ...prev.data_source,
                            columns: headers.map((header) => ({
                                name: header,
                                type: "string",
                                role: "data",
                                required: false,
                            })),
                        },
                    }));
                } catch (err) {
                    setError(`Failed to parse CSV file: ${err.message}`);
                }
            };

            reader.onerror = () => {
                setError("Failed to read CSV file");
            };

            reader.readAsText(file);
        } catch (err) {
            setError(`Failed to process the file: ${err.message}`);
        } finally {
            setIsUploading(false);
        }
    }, []);

    return { handleFileUpload, isUploading, csvHeaders, error };
};

const useTraining = (configId) => {
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [error, setError] = useState("");

    const monitorTraining = useCallback(async (configId) => {
        const checkStatus = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/config/status/${configId}`);
                const data = await response.json();
                setTrainingStatus(data);

                if (data.status === ModelStatus.PROCESSING || data.status === ModelStatus.QUEUED) {
                    setTimeout(checkStatus, 5000);
                }
            } catch (err) {
                setError("Failed to check training status");
            }
        };

        checkStatus();
    }, []);

    useEffect(() => {
        if (configId) {
            monitorTraining(configId);
        }
    }, [configId, monitorTraining]);

    return { trainingStatus, error, monitorTraining };
};

const useSearch = () => {
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [error, setError] = useState("");

    const performSearch = async (query, configId) => {
        try {
            setIsSearching(true);
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    config_id: configId || "latest",
                    max_items: 10,
                }),
            });

            if (!response.ok) {
                throw new Error("Search failed");
            }

            const data = await response.json();
            setSearchResults(data.results);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsSearching(false);
        }
    };

    return { performSearch, searchResults, isSearching, error };
};

// Status Badge Component
const StatusBadge = ({ status }) => {
    const getStatusColor = (status) => {
        switch (status) {
            case ModelStatus.COMPLETED:
                return "bg-green-100 text-green-800";
            case ModelStatus.FAILED:
                return "bg-red-100 text-red-800";
            case ModelStatus.PROCESSING:
                return "bg-blue-100 text-blue-800";
            case ModelStatus.QUEUED:
            case ModelStatus.PENDING:
                return "bg-yellow-100 text-yellow-800";
            case ModelStatus.CANCELED:
                return "bg-gray-100 text-gray-800";
            default:
                return "bg-gray-100 text-gray-800";
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case ModelStatus.COMPLETED:
                return <CheckCircle2 className="w-4 h-4" />;
            case ModelStatus.FAILED:
                return <XCircle className="w-4 h-4" />;
            case ModelStatus.PROCESSING:
                return <Loader2 className="w-4 h-4 animate-spin" />;
            case ModelStatus.QUEUED:
            case ModelStatus.PENDING:
                return <Clock className="w-4 h-4" />;
            case ModelStatus.CANCELED:
                return <AlertTriangle className="w-4 h-4" />;
            default:
                return null;
        }
    };

    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-sm font-medium ${getStatusColor(status)}`}>
            {getStatusIcon(status)}
            {status.toLowerCase()}
        </span>
    );
};

// Model List Component
const ModelList = ({ onSelect, selectedConfigId }) => {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        const fetchModels = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/config`);
                if (!response.ok) throw new Error("Failed to fetch models");
                const data = await response.json();
                setModels(data.configs);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchModels();
    }, []);

    if (loading) return <div className="p-4">Loading models...</div>;
    if (error) return <div className="p-4 text-red-500">{error}</div>;

    return (
        <div className="space-y-4">
            {models.map((model) => (
                <div
                    key={model._id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedConfigId === model._id ? "border-blue-500 bg-blue-50" : "hover:bg-gray-50"
                    }`}
                    onClick={() => onSelect(model._id)}
                >
                    <div className="flex justify-between items-center mb-2">
                        <h3 className="font-semibold">{model.name}</h3>
                        <StatusBadge status={model.status} />
                    </div>
                    <p className="text-sm text-gray-600">{model.description}</p>
                    <div className="mt-2 text-xs text-gray-500">
                        Created: {new Date(model.created_at).toLocaleString()}
                    </div>
                </div>
            ))}
        </div>
    );
};

// Search Box Component
const SearchBox = ({ selectedConfigId, onSearch, loading }) => {
    const [query, setQuery] = useState("");

    const handleSearch = (e) => {
        e.preventDefault();
        onSearch(query);
    };

    return (
        <form onSubmit={handleSearch} className="space-y-4">
            <div className="flex gap-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter search query..."
                    className="flex-1 p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
                <button
                    type="submit"
                    disabled={loading || !query}
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 flex items-center"
                >
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                </button>
            </div>
        </form>
    );
};

// Search Results Component
const SearchResults = ({ results }) => {
    if (!results?.length) return null;

    return (
        <div className="space-y-4">
            {results.map((result, index) => (
                <div key={index} className="p-4 border rounded-lg">
                    <h3 className="font-semibold">{result.name}</h3>
                    <p className="text-sm text-gray-600 mt-1">{result.description}</p>
                    <div className="flex justify-between items-center mt-2">
                        <span className="text-sm text-gray-500">{result.category}</span>
                        <span className="text-sm font-medium">
                            Score: {(result.score * 100).toFixed(1)}%
                        </span>
                    </div>
                    {Object.keys(result.metadata).length > 0 && (
                        <div className="mt-2 text-sm text-gray-500">
                            {Object.entries(result.metadata).map(([key, value]) => (
                                <div key={key}>
                                    {key}: {value}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
};

export default function ProductSearchConfig() {
    const {
        config,
        setConfig,
        submitConfig,
        isSubmitting,
        error: configError,
        configId,
    } = useModelConfig();

    const { handleFileUpload, isUploading, csvHeaders, error: uploadError } = useFileUpload();
    const { trainingStatus, error: trainingError } = useTraining(configId);
    const { performSearch, searchResults, isSearching, error: searchError } = useSearch();

    const [selectedConfigId, setSelectedConfigId] = useState(null);
    const [step, setStep] = useState(1);
    const [csvFile, setCsvFile] = useState(null);

    const handleFileUploadWrapper = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            handleFileUpload(file, setConfig);
            setCsvFile(file);
        }
    };

    const handleSubmit = async () => {
        try {
            if (!csvFile) {
                throw new Error("Please upload a CSV file first");
            }
            await submitConfig(csvFile);
        } catch (err) {
            console.error("Submission error:", err);
        }
    };

    const handleSearch = (query) => {
        performSearch(query, selectedConfigId);
    };

    return (
        <div className="max-w-4xl mx-auto p-6 space-y-8">
            <h1 className="text-2xl font-bold">Product Search Configuration</h1>

            {/* Error Display */}
            {(configError || uploadError || trainingError || searchError) && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center text-red-700">
                    <AlertCircle className="h-4 w-4 mr-2" />
                    <p>{configError || uploadError || trainingError || searchError}</p>
                </div>
            )}

            {/* Success Message */}
            {trainingStatus?.status === ModelStatus.COMPLETED && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center text-green-700">
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    <p>Model trained successfully! Ready for search.</p>
                </div>
            )}

            {/* Main Content Area */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Left Column - Configuration Form */}
                <div className="space-y-6">
                    <div className="border rounded-lg p-6">
                        <h2 className="text-xl font-semibold mb-4">Create New Model</h2>
                        
                        {/* Basic Info */}
                        <div className="space-y-4 mb-6">
                            <div>
                                <label className="block text-sm font-medium mb-1">Model Name</label>
                                <input
                                    type="text"
                                    value={config.name}
                                    onChange={(e) => setConfig(prev => ({ ...prev, name: e.target.value }))}
                                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                    placeholder="Enter model name"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1">Description</label>
                                <textarea
                                    value={config.description}
                                    onChange={(e) => setConfig(prev => ({ ...prev, description: e.target.value }))}
                                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                    placeholder="Enter description"
                                    rows={3}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1">Training Mode</label>
                                <select
                                    value={config.mode}
                                    onChange={(e) => setConfig(prev => ({ ...prev, mode: e.target.value }))}
                                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                >
                                    <option value="replace">Replace (New Model)</option>
                                    <option value="append">Append (Update Existing)</option>
                                </select>
                            </div>
                            {config.mode === "append" && (
                                <div>
                                    <label className="block text-sm font-medium mb-1">Previous Model</label>
                                    <ModelList
                                        onSelect={(id) => setConfig(prev => ({ ...prev, previous_version: id }))}
                                        selectedConfigId={config.previous_version}
                                    />
                                </div>
                            )}
                        </div>

                        {/* File Upload */}
                        <div className="space-y-4 mb-6">
                            <label className="block text-sm font-medium mb-1">Upload Data</label>
                            <div
                                className="border-2 border-dashed rounded-lg p-8 text-center hover:border-blue-500 transition-colors"
                                onDragOver={(e) => e.preventDefault()}
                                onDrop={(e) => {
                                    e.preventDefault();
                                    const file = e.dataTransfer.files[0];
                                    if (file) handleFileUploadWrapper({ target: { files: [file] } });
                                }}
                            >
                                <div className="flex flex-col items-center space-y-4">
                                    <Upload className="h-12 w-12 text-gray-400" />
                                    <div className="space-y-2">
                                        <h3 className="font-semibold">Upload CSV file</h3>
                                        <p className="text-sm text-gray-500">Drag and drop or click to select</p>
                                    </div>
                                    <label className="cursor-pointer">
                                        <input
                                            type="file"
                                            className="hidden"
                                            accept=".csv"
                                            onChange={handleFileUploadWrapper}
                                            disabled={isUploading}
                                        />
                                        <span className="px-4 py-2 border rounded hover:bg-gray-50 inline-block">
                                            {isUploading ? "Uploading..." : "Select File"}
                                        </span>
                                    </label>
                                </div>
                            </div>
                            {csvFile && (
                                <div className="bg-green-50 border border-green-200 rounded p-4 flex items-center">
                                    <CheckCircle2 className="h-4 w-4 text-green-500 mr-2" />
                                    <span>File loaded: {csvFile.name}</span>
                                </div>
                            )}
                        </div>

                        {/* Column Mapping */}
                        {csvHeaders.length > 0 && (
                            <div className="space-y-4">
                                <h3 className="font-semibold">Column Mapping</h3>
                                <div className="grid gap-4">
                                    {["id", "name", "description", "category"].map((field) => (
                                        <div key={field}>
                                            <label className="block text-sm font-medium mb-1">
                                                {field.charAt(0).toUpperCase() + field.slice(1)} Column
                                            </label>
                                            <select
                                                value={config.schema_mapping[`${field}_column`]}
                                                onChange={(e) => setConfig(prev => ({
                                                    ...prev,
                                                    schema_mapping: {
                                                        ...prev.schema_mapping,
                                                        [`${field}_column`]: e.target.value
                                                    }
                                                }))}
                                                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                            >
                                                <option value="">Select column</option>
                                                {csvHeaders.map(header => (
                                                    <option key={header} value={header}>{header}</option>
                                                ))}
                                            </select>
                                        </div>
                                    ))}
                                </div>

                                {/* Custom Columns */}
                                <div className="border rounded-lg p-4 space-y-4">
                                    <div className="flex justify-between items-center">
                                        <h3 className="font-semibold">Custom Columns</h3>
                                        <button
                                            onClick={() => setConfig(prev => ({
                                                ...prev,
                                                schema_mapping: {
                                                    ...prev.schema_mapping,
                                                    custom_columns: [
                                                        ...prev.schema_mapping.custom_columns,
                                                        { user_column: "", standard_column: "", role: "metadata" }
                                                    ]
                                                }
                                            }))}
                                            className="px-3 py-1 text-sm border rounded hover:bg-gray-50"
                                        >
                                            Add Column
                                        </button>
                                    </div>

                                    {config.schema_mapping.custom_columns.map((col, index) => (
                                        <div key={index} className="flex gap-2 items-center">
                                            <select
                                                value={col.user_column}
                                                onChange={(e) => {
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
                                                className="flex-1 p-2 border rounded"
                                            >
                                                <option value="">Select column</option>
                                                {csvHeaders.map(header => (
                                                    <option key={header} value={header}>{header}</option>
                                                ))}
                                            </select>
                                            <input
                                                type="text"
                                                placeholder="Standard name"
                                                value={col.standard_column}
                                                onChange={(e) => {
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
                                                className="flex-1 p-2 border rounded"
                                            />
                                            <select
                                                value={col.role}
                                                onChange={(e) => {
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
                                                className="w-32 p-2 border rounded"
                                            >
                                                <option value="metadata">Metadata</option>
                                                <option value="training">Training</option>
                                            </select>
                                            <button
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
                                                className="p-2 text-red-500 hover:text-red-600"
                                            >
                                                <XCircle className="h-4 w-4" />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Submit Button */}
                        <div className="mt-6">
                            <button
                                onClick={handleSubmit}
                                disabled={isSubmitting || !csvFile}
                                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 flex items-center justify-center"
                            >
                                {isSubmitting ? (
                                    <>
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Processing...
                                    </>
                                ) : (
                                    "Start Training"
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right Column - Models and Search */}
                <div className="space-y-6">
                    {/* Available Models */}
                    <div className="border rounded-lg p-6">
                        <h2 className="text-xl font-semibold mb-4">Available Models</h2>
                        <ModelList
                            onSelect={setSelectedConfigId}
                            selectedConfigId={selectedConfigId}
                        />
                    </div>

                    {/* Search Box */}
                    {selectedConfigId && (
                        <div className="border rounded-lg p-6">
                            <h2 className="text-xl font-semibold mb-4">Search Products</h2>
                            <SearchBox
                                selectedConfigId={selectedConfigId}
                                onSearch={handleSearch}
                                loading={isSearching}
                            />
                            <div className="mt-4">
                                <SearchResults results={searchResults} />
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Training Status */}
            {configId && trainingStatus && (
                <div className="border rounded-lg p-6 mt-6">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-semibold">Training Status</h2>
                        <StatusBadge status={trainingStatus.status} />
                    </div>
                    <div className="space-y-2">
                        <div className="text-sm text-gray-600">
                            {trainingStatus.status === ModelStatus.COMPLETED && "Training completed successfully"}
                            {trainingStatus.status === ModelStatus.PROCESSING && "Training in progress..."}
                            {trainingStatus.status === ModelStatus.FAILED && (
                                <div className="text-red-500">
                                    Training failed: {trainingStatus.error}
                                </div>
                            )}
                        </div>
                        {trainingStatus.training_stats && (
                            <div className="text-sm text-gray-600">
                                <div>Processed Records: {trainingStatus.training_stats.processed_records}</div>
                                {trainingStatus.training_stats.training_accuracy && (
                                    <div>Training Accuracy: {(trainingStatus.training_stats.training_accuracy * 100).toFixed(2)}%</div>
                                )}
                                {trainingStatus.training_stats.progress && (
                                    <div className="relative pt-1">
                                        <div className="overflow-hidden h-2 text-xs flex rounded bg-blue-200">
                                            <div
                                                style={{ width: `${trainingStatus.training_stats.progress}%` }}
                                                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}