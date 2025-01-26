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
    PENDING: "PENDING",
    QUEUED: "QUEUED",
    PROCESSING: "PROCESSING",
    COMPLETED: "COMPLETED",
    FAILED: "FAILED",
    CANCELED: "CANCELED",
};

const useModelConfig = () => {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState("");
    const [versionId, setVersionId] = useState(null);
    const [config, setConfig] = useState({
        name: "",
        description: "",
        data_source: {
            type: "csv",
            location: "",
            columns: [],
        },
        schema_mapping: {
            id_column: "",
            name_column: "",
            description_column: "",
            category_column: "",
            custom_columns: [],
        },
        training_config: {
            model_type: "transformer",
            embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
            batch_size: 128,
            max_tokens: 512,
        },
    });

    const submitConfig = async (file) => {
        setIsSubmitting(true);
        setError("");

        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("config", JSON.stringify(config));

            const response = await fetch(`${API_BASE_URL}/config/create`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Failed to create configuration");
            }

            const data = await response.json();
            setVersionId(data.data.version_id);
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
        versionId,
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
            // Read file locally to get headers
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

const useTraining = () => {
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [error, setError] = useState("");

    const startTraining = async (config, file) => {
        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("config", JSON.stringify(config));

            const response = await fetch(`${API_BASE_URL}/config/create`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Failed to create configuration and start training");
            }

            const data = await response.json();
            monitorTraining(data.data.version_id);
            return data;
        } catch (err) {
            setError(err.message);
            throw err;
        }
    };

    const monitorTraining = useCallback(async (versionId) => {
        const checkStatus = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/training/status/${versionId}`);
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

    return { startTraining, trainingStatus, error, monitorTraining  };
};

const useSearch = () => {
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [error, setError] = useState("");

    const performSearch = async (query, versionId = "latest") => {
        try {
            setIsSearching(true);
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    version: versionId,
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
            <div key={index} className="p-4 border rounded-lg hover:shadow-md transition-shadow duration-200 bg-white">
                <h4 className="font-semibold text-lg">{result.name}</h4>
                <p className="text-sm text-gray-600 mb-2">{result.description}</p>
                <div className="flex justify-between items-center text-sm text-gray-500">
                    <span className="bg-gray-100 px-2 py-1 rounded-full text-xs">{result.category}</span>
                    <span className="font-medium">Match: {(result.score * 100).toFixed(1)}%</span>
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
                    <button onClick={onCancel} className="px-4 py-2 border rounded hover:bg-gray-50">
                        Cancel
                    </button>
                    <button onClick={onConfirm} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
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
        if (!config.name.trim()) newErrors.name = "Name is required";
        if (!config.description.trim()) newErrors.description = "Description is required";
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
                        onChange={(e) => setConfig((prev) => ({ ...prev, name: e.target.value }))}
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
                        onChange={(e) => setConfig((prev) => ({ ...prev, description: e.target.value }))}
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
const CsvUploadStep = ({ onFileUpload, csvHeaders, csvFile, onBack, onNext, isUploading, error }) => {
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

    const handleDrop = useCallback(
        (e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(false);

            const files = e.dataTransfer.files;
            if (files && files[0]) {
                const file = files[0];
                if (file.type === "text/csv" || file.name.endsWith(".csv")) {
                    onFileUpload({ target: { files: [file] } });
                } else {
                    // Display error for non-CSV files
                    alert("Please upload a CSV file");
                }
            }
        },
        [onFileUpload]
    );

    const handleFileInput = useCallback(
        (e) => {
            const file = e.target.files?.[0];
            if (file) {
                if (file.type === "text/csv" || file.name.endsWith(".csv")) {
                    onFileUpload(e);
                } else {
                    // Display error for non-CSV files
                    alert("Please upload a CSV file");
                }
            }
        },
        [onFileUpload]
    );

    return (
        <div className="space-y-6 p-6">
            <h2 className="text-xl font-semibold">Upload Product Data</h2>
            <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${dragActive ? "border-blue-500 bg-blue-50" : "hover:border-blue-500"
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
                        <p className="text-sm text-gray-500">Drag and drop or click to select</p>
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
                            className={`px-4 py-2 border rounded ${isUploading ? "bg-gray-100" : "hover:bg-gray-50"}`}
                            disabled={isUploading}
                            onClick={() => document.querySelector('input[type="file"]').click()}
                        >
                            {isUploading ? (
                                <span className="flex items-center">
                                    <Loader2 className="animate-spin mr-2 h-4 w-4" />
                                    Uploading...
                                </span>
                            ) : (
                                "Select File"
                            )}
                        </button>
                    </label>
                </div>
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 rounded p-4 flex items-center text-red-700">
                    <AlertCircle className="h-4 w-4 mr-2" />
                    <span>{error}</span>
                </div>
            )}

            {csvFile && !error && (
                <div className="bg-green-50 border border-green-200 rounded p-4 flex items-center">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mr-2" />
                    <span>Successfully loaded: {csvFile.name}</span>
                </div>
            )}

            {csvHeaders.length > 0 && (
                <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-2">Detected Columns</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                        {csvHeaders.map((header) => (
                            <div key={header} className="px-3 py-1 bg-gray-100 rounded-full text-sm text-center">
                                {header}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="flex justify-between pt-4">
                <button className="px-4 py-2 border rounded hover:bg-gray-50" onClick={onBack}>
                    Back
                </button>
                <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                    onClick={onNext}
                    disabled={!csvFile || isUploading || error}
                >
                    Next
                </button>
            </div>
        </div>
    );
};

// Column Mapping Step
const ColumnMappingStep = ({ config, setConfig, csvHeaders, onBack, onSubmit, loading, error }) => {
    const [errors, setErrors] = useState({});

    const validate = () => {
        const newErrors = {};
        if (!config.schema_mapping.id_column) newErrors.id = "ID column is required";
        if (!config.schema_mapping.name_column) newErrors.name = "Name column is required";
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = () => {
        if (validate()) onSubmit();
    };

    const handleCustomColumnAdd = () => {
        setConfig((prev) => ({
            ...prev,
            schema_mapping: {
                ...prev.schema_mapping,
                custom_columns: [
                    ...prev.schema_mapping.custom_columns,
                    { user_column: "", standard_column: "", role: "metadata" },
                ],
            },
        }));
    };

    return (
        <div className="space-y-6 p-6">
            <h2 className="text-xl font-semibold">Column Mapping</h2>
            <div className="space-y-6">
                <div className="grid gap-4">
                    {["id", "name", "description", "category"].map((field) => (
                        <FormField
                            key={field}
                            label={`${field.charAt(0).toUpperCase() + field.slice(1)} Column`}
                            tooltip={`Select the column that contains ${field} information`}
                            error={errors[field]}
                        >
                            <select
                                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                value={config.schema_mapping[`${field}_column`]}
                                onChange={(e) =>
                                    setConfig((prev) => ({
                                        ...prev,
                                        schema_mapping: {
                                            ...prev.schema_mapping,
                                            [`${field}_column`]: e.target.value,
                                        },
                                    }))
                                }
                            >
                                <option value="">Select column</option>
                                {csvHeaders.map((header) => (
                                    <option key={header} value={header}>
                                        {header}
                                    </option>
                                ))}
                            </select>
                        </FormField>
                    ))}
                </div>

                <div className="border rounded-lg p-4 space-y-4">
                    <div className="flex justify-between items-center">
                        <h3 className="font-semibold">Custom Columns</h3>
                        <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50" onClick={handleCustomColumnAdd}>
                            Add Column
                        </button>
                    </div>

                    {config.schema_mapping.custom_columns.map((col, index) => (
                        <div key={index} className="flex gap-2 items-center">
                            <select
                                className="flex-1 p-2 border rounded"
                                value={col.user_column}
                                onChange={(e) => {
                                    const newColumns = [...config.schema_mapping.custom_columns];
                                    newColumns[index].user_column = e.target.value;
                                    setConfig((prev) => ({
                                        ...prev,
                                        schema_mapping: {
                                            ...prev.schema_mapping,
                                            custom_columns: newColumns,
                                        },
                                    }));
                                }}
                            >
                                <option value="">Select column</option>
                                {csvHeaders.map((header) => (
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
                                onChange={(e) => {
                                    const newColumns = [...config.schema_mapping.custom_columns];
                                    newColumns[index].standard_column = e.target.value;
                                    setConfig((prev) => ({
                                        ...prev,
                                        schema_mapping: {
                                            ...prev.schema_mapping,
                                            custom_columns: newColumns,
                                        },
                                    }));
                                }}
                            />
                            <select
                                className="p-2 border rounded"
                                value={col.role}
                                onChange={(e) => {
                                    const newColumns = [...config.schema_mapping.custom_columns];
                                    newColumns[index].role = e.target.value;
                                    setConfig((prev) => ({
                                        ...prev,
                                        schema_mapping: {
                                            ...prev.schema_mapping,
                                            custom_columns: newColumns,
                                        },
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
                                    setConfig((prev) => ({
                                        ...prev,
                                        schema_mapping: {
                                            ...prev.schema_mapping,
                                            custom_columns: newColumns,
                                        },
                                    }));
                                }}
                            >
                                ×
                            </button>
                        </div>
                    ))}
                </div>

                <div className="flex justify-between pt-4">
                    <button className="px-4 py-2 border rounded hover:bg-gray-50" onClick={onBack}>
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
                            "Start Training"
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
    loading,
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
                {modelVersions?.map((version) => (
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
                        if (e.key === "Enter" && !loading) {
                            handleSearch();
                        }
                    }}
                />
                <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 flex items-center"
                    onClick={handleSearch}
                    disabled={loading || !searchQuery}
                >
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                </button>
            </div>
        </div>
    </div>
);

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
        <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-sm font-medium ${getStatusColor(status)}`}
        >
            {getStatusIcon(status)}
            {status.toLowerCase()}
        </span>
    );
};

// Status Description Component
const StatusDescription = ({ status, error }) => {
    const getStatusMessage = (status) => {
        switch (status) {
            case ModelStatus.PENDING:
                return "Model version created, waiting to be queued";
            case ModelStatus.QUEUED:
                return "Training job queued, waiting to be processed";
            case ModelStatus.PROCESSING:
                return "Model training in progress";
            case ModelStatus.COMPLETED:
                return "Training completed successfully";
            case ModelStatus.FAILED:
                return `Training failed: ${error || "Unknown error"}`;
            case ModelStatus.CANCELED:
                return "Training was canceled";
            default:
                return "Unknown status";
        }
    };

    return <div className="text-sm text-gray-600">{getStatusMessage(status)}</div>;
};

// Progress Tracker Component
const TrainingProgress = ({ status, error }) => {
    const steps = [
        { id: 1, name: "Created", status: [ModelStatus.PENDING] },
        { id: 2, name: "Queued", status: [ModelStatus.QUEUED] },
        { id: 3, name: "Processing", status: [ModelStatus.PROCESSING] },
        { id: 4, name: "Completed", status: [ModelStatus.COMPLETED, ModelStatus.FAILED, ModelStatus.CANCELED] },
    ];

    const getCurrentStep = () => {
        return steps.findIndex((step) => step.status.includes(status)) + 1;
    };

    return (
        <div className="py-4">
            <div className="flex items-center justify-between">
                {steps.map((step, index) => (
                    <React.Fragment key={step.id}>
                        <div className="flex flex-col items-center">
                            <div
                                className={`w-8 h-8 rounded-full flex items-center justify-center ${getCurrentStep() > step.id
                                    ? "bg-blue-600"
                                    : getCurrentStep() === step.id
                                        ? "bg-blue-200"
                                        : "bg-gray-200"
                                    }`}
                            >
                                {getCurrentStep() > step.id ? (
                                    <CheckCircle2 className="w-5 h-5 text-white" />
                                ) : (
                                    <span className="text-sm">{step.id}</span>
                                )}
                            </div>
                            <div className="text-xs mt-1">{step.name}</div>
                        </div>
                        {index < steps.length - 1 && (
                            <div className={`flex-1 h-0.5 ${getCurrentStep() > step.id + 1 ? "bg-blue-600" : "bg-gray-200"}`} />
                        )}
                    </React.Fragment>
                ))}
            </div>
            <StatusDescription status={status} error={error} />
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
        versionId,
    } = useModelConfig();
    const { handleFileUpload, isUploading, csvHeaders, error: uploadError } = useFileUpload();
    const { trainingStatus, error: trainingError, monitorTraining } = useTraining();
    const { performSearch, searchResults, isSearching, error: searchError } = useSearch();

    const [selectedVersion, setSelectedVersion] = useState("latest");
    const [modelVersions, setModelVersions] = useState([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [step, setStep] = useState(1);
    const [csvFile, setCsvFile] = useState(null);
    const [showNavigationPrompt, setShowNavigationPrompt] = useState(false);
    const [pendingNavigation, setPendingNavigation] = useState(null);
    const [fileUploadError, setFileUploadError] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        fetchModelVersions();
    }, []);

    useEffect(() => {
        if (versionId) {
            monitorTraining(versionId);
        }
    }, [versionId, monitorTraining]);

    const handleSubmit = async () => {
        try {
            if (!csvFile) {
                setError("Please upload a CSV file first");
                return;
            }

            setLoading(true);
            const result = await submitConfig(csvFile);

            // Update model versions list after successful training start
            fetchModelVersions();

            // Show success message
            setError("Training started successfully!");
        } catch (err) {
            setFileUploadError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const fetchModelVersions = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/model/versions`);
            if (response.ok) {
                const data = await response.json();
                setModelVersions(data);
            }
        } catch (err) {
            console.error("Failed to fetch model versions:", err);
        }
    };

    const handleNavigationAttempt = (targetStep) => {
        if (isUploading || trainingStatus?.status === ModelStatus.PROCESSING) {
            setShowNavigationPrompt(true);
            setPendingNavigation(targetStep);
            return false;
        }
        return true;
    };

    const getError = () => {
        return configError || uploadError || trainingError || searchError;
    };

    const handleNavigationConfirm = () => {
        setShowNavigationPrompt(false);
        if (pendingNavigation !== null) {
            setStep(pendingNavigation);
            setPendingNavigation(null);
        }
    };

    const handleFileUploadWrapper = (e) => {
        setFileUploadError(null); // Reset error on new file selection
        const file = e.target.files?.[0];
        if (file) {
            handleFileUpload(file, setConfig);
            setCsvFile(file);
        }
    };

    const handleSearch = () => {
        performSearch(searchQuery, selectedVersion);
    };

    return (
        <div className="max-w-4xl mx-auto p-6 space-y-8">
            <h1 className="text-2xl font-bold mb-6">Product Search Configuration</h1>

            <NavigationPrompt
                isOpen={showNavigationPrompt}
                onConfirm={handleNavigationConfirm}
                onCancel={() => setShowNavigationPrompt(false)}
            />

            {getError() && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center text-red-700">
                    <AlertCircle className="h-4 w-4 mr-2" />
                    <p>{getError()}</p>
                </div>
            )}

            {trainingStatus?.status === ModelStatus.COMPLETED && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center text-green-700">
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    <p>Training completed successfully!</p>
                </div>
            )}

            <SearchBox
                selectedVersion={selectedVersion}
                setSelectedVersion={setSelectedVersion}
                modelVersions={modelVersions}
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                handleSearch={handleSearch}
                loading={isSearching}
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
                        onFileUpload={handleFileUploadWrapper}
                        csvHeaders={csvHeaders}
                        csvFile={csvFile}
                        onBack={() => setStep(1)}
                        onNext={() => setStep(3)}
                        isUploading={isUploading}
                        error={uploadError}
                    />
                )}

                {step === 3 && (
                    <ColumnMappingStep
                        config={config}
                        setConfig={setConfig}
                        csvHeaders={csvHeaders}
                        onBack={() => handleNavigationAttempt(2) && setStep(2)}
                        onSubmit={handleSubmit}
                        loading={isSubmitting || trainingStatus?.status === ModelStatus.PROCESSING}
                        error={configError}
                    />
                )}
            </div>

            {(trainingStatus || versionId) && (
                <div className="border rounded-lg p-6 space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="font-semibold">Training Status</h3>
                        <StatusBadge status={trainingStatus?.status || ModelStatus.PENDING} />
                    </div>

                    <TrainingProgress
                        status={trainingStatus?.status || ModelStatus.PENDING}
                        error={trainingStatus?.error}
                    />

                    {trainingStatus?.error && (
                        <p className="text-sm text-red-500">{trainingStatus.error}</p>
                    )}
                </div>
            )}

            {searchResults?.length > 0 && <SearchResults results={searchResults} />}
        </div>
    );
}