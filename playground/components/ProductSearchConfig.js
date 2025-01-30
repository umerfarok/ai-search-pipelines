import React, { useState, useCallback, useEffect } from "react";
import {
    AlertCircle, Upload, CheckCircle2, XCircle,
    Loader2, Clock, AlertTriangle, ChevronDown, ChevronUp,
    Info, RefreshCw
} from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Tooltip } from "@/components/ui/tooltip";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

const ModelStatus = {
    PENDING: "pending",
    QUEUED: "queued",
    PROCESSING: "processing",
    COMPLETED: "completed",
    FAILED: "failed",
    CANCELED: "canceled"
};

// Updated model configurations
const AVAILABLE_LLM_MODELS = {
    "distilgpt2": {
        name: "DistilGPT-2",
        description: "Lightweight model optimized for product search",
        tags: ["fast", "efficient"]
    },
    "all-minilm-l6": {
        name: "all-MiniLM-L6-v2",
        description: "Efficient embedding model for semantic search",
        tags: ["semantic", "fast"]
    }
};

// Reusable UI Components
const FormSection = ({ title, children, description, className = "" }) => (
    <div className={`mt-6 ${className}`}>
        <div className="flex items-center gap-2 mb-4">
            <h3 className="text-sm font-medium text-gray-700">{title}</h3>
            {description && (
                <Tooltip content={description}>
                    <Info className="w-4 h-4 text-gray-400" />
                </Tooltip>
            )}
        </div>
        {children}
    </div>
);

const Input = ({ label, error, ...props }) => (
    <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
        <input
            {...props}
            className={`w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 
                ${error ? 'border-red-300 bg-red-50' : 'border-gray-300'}`}
        />
        {error && <p className="mt-1 text-sm text-red-600">{error}</p>}
    </div>
);

const Select = ({ label, options, error, ...props }) => (
    <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
        <select
            {...props}
            className={`w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                ${error ? 'border-red-300 bg-red-50' : 'border-gray-300'}`}
        >
            <option value="">Select {label.toLowerCase()}</option>
            {options.map(({ value, label, description }) => (
                <option key={value} value={value} title={description}>{label}</option>
            ))}
        </select>
        {error && <p className="mt-1 text-sm text-red-600">{error}</p>}
    </div>
);

const Card = ({ title, subtitle, children, className = "" }) => (
    <div className={`bg-white border border-gray-200 rounded-lg shadow-sm p-6 ${className}`}>
        {title && (
            <div className="mb-6">
                <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
                {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
            </div>
        )}
        {children}
    </div>
);

// Custom Hooks with improved error handling and validation
const useModelConfig = () => {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState("");
    const [validationErrors, setValidationErrors] = useState({});
    const [configId, setConfigId] = useState(null);
    const [config, setConfig] = useState({
        name: "",
        description: "",
        mode: "replace",
        previous_version: "",
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
            embedding_model: "all-minilm-l6",
            batch_size: 32,
            max_tokens: 512,
            validation_split: 0.2,
            llm_model: "distilgpt2",
            training_params: {},
        },
    });

    const validateConfig = () => {
        const errors = {};
        if (!config.name) errors.name = "Name is required";
        if (!config.schema_mapping.id_column) errors.id_column = "ID column is required";
        if (!config.schema_mapping.name_column) errors.name_column = "Name column is required";

        setValidationErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const submitConfig = async (file) => {
        try {
            setIsSubmitting(true);
            setError("");

            if (!validateConfig()) {
                throw new Error("Please fix validation errors before submitting");
            }

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
        validationErrors
    };
};

const useFileUpload = () => {
    const [isUploading, setIsUploading] = useState(false);
    const [csvHeaders, setCsvHeaders] = useState([]);
    const [error, setError] = useState("");
    const [uploadProgress, setUploadProgress] = useState(0);

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
        setUploadProgress(0);

        try {
            const reader = new FileReader();

            reader.onprogress = (event) => {
                if (event.lengthComputable) {
                    setUploadProgress((event.loaded / event.total) * 100);
                }
            };

            reader.onload = (event) => {
                try {
                    const text = event.target.result;
                    const lines = text.split("\n");

                    if (lines.length === 0) {
                        throw new Error("The CSV file is empty");
                    }

                    const headers = lines[0].trim().split(",").map((h) => h.trim());

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

                    setUploadProgress(100);

                } catch (err) {
                    setError(`Failed to parse CSV file: ${err.message}`);
                }
            };

            reader.onerror = () => {
                setError("Failed to read CSV file");
                setUploadProgress(0);
            };

            reader.readAsText(file);

        } catch (err) {
            setError(`Failed to process the file: ${err.message}`);
            setUploadProgress(0);
        } finally {
            setIsUploading(false);
        }
    }, []);

    return {
        handleFileUpload,
        isUploading,
        csvHeaders,
        error,
        uploadProgress
    };
};

const useTraining = (configId) => {
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [error, setError] = useState("");
    const [retryCount, setRetryCount] = useState(0);
    const maxRetries = 3;

    const monitorTraining = useCallback(async (configId) => {
        const checkStatus = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/config/status/${configId}`);

                if (!response.ok) {
                    throw new Error("Failed to fetch status");
                }

                const data = await response.json();
                setTrainingStatus(data);

                if (data.status === ModelStatus.PROCESSING || data.status === ModelStatus.QUEUED) {
                    setTimeout(checkStatus, 5000);
                }

                // Reset retry count on successful request
                setRetryCount(0);

            } catch (err) {
                setError("Failed to check training status");

                if (retryCount < maxRetries) {
                    setRetryCount(prev => prev + 1);
                    setTimeout(checkStatus, 5000 * (retryCount + 1));
                }
            }
        };

        checkStatus();
    }, [configId, retryCount]);

    useEffect(() => {
        if (configId) monitorTraining(configId);
    }, [configId, monitorTraining]);

    const retryTraining = useCallback(() => {
        setError("");
        setRetryCount(0);
        monitorTraining(configId);
    }, [configId, monitorTraining]);

    return { trainingStatus, error, retryTraining };
};

// Status Badge Component with improved visuals
const StatusBadge = ({ status, showIcon = true }) => {
    const statusConfig = {
        [ModelStatus.COMPLETED]: {
            color: "green",
            icon: CheckCircle2,
            text: "Completed"
        },
        [ModelStatus.FAILED]: {
            color: "red",
            icon: XCircle,
            text: "Failed"
        },
        [ModelStatus.PROCESSING]: {
            color: "blue",
            icon: Loader2,
            text: "Processing",
            animate: true
        },
        [ModelStatus.QUEUED]: {
            color: "yellow",
            icon: Clock,
            text: "Queued"
        },
        [ModelStatus.PENDING]: {
            color: "yellow",
            icon: Clock,
            text: "Pending"
        },
        [ModelStatus.CANCELED]: {
            color: "gray",
            icon: AlertTriangle,
            text: "Canceled"
        }
    }[status] || {
        color: "gray",
        icon: AlertCircle,
        text: "Unknown"
    };

    const Icon = statusConfig.icon;

    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-sm font-medium
            bg-${statusConfig.color}-100 text-${statusConfig.color}-800`}>
            {showIcon && (
                <Icon className={`w-4 h-4 ${statusConfig.animate ? 'animate-spin' : ''}`} />
            )}
            {statusConfig.text}
        </span>
    );
};

// Model List Component with improved UI
const ModelList = ({ onSelect, selectedConfigId }) => {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [sortField, setSortField] = useState("created_at");
    const [sortOrder, setSortOrder] = useState("desc");

    const fetchModels = useCallback(async () => {
        try {
            setLoading(true);
            const response = await fetch(`${API_BASE_URL}/config`);

            if (!response.ok) {
                throw new Error("Failed to fetch models");
            }

            const data = await response.json();
            setModels(data.configs);

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchModels();
    }, [fetchModels]);

    const sortedModels = [...(models || [])].sort((a, b) => {
        if (sortField === "created_at") {
            return sortOrder === "desc"
                ? new Date(b.created_at) - new Date(a.created_at)
                : new Date(a.created_at) - new Date(b.created_at);
        }
        return 0;
    });

    if (loading) {
        return (
            <div className="p-4 flex items-center justify-center">
                <Loader2 className="w-6 h-6 text-blue-500 animate-spin mr-2" />
                <span>Loading models...</span>
            </div>
        );
    }

    if (error) {
        return (
            <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
                <button
                    onClick={fetchModels}
                    className="mt-2 flex items-center text-sm text-blue-600 hover:text-blue-800"
                >
                    <RefreshCw className="w-4 h-4 mr-1" />
                    Retry
                </button>
            </Alert>
        );
    }

    return (
        <div className="space-y-4">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">Available Models</h3>
                <button
                    onClick={() => setSortOrder(prev => prev === "desc" ? "asc" : "desc")}
                    className="flex items-center text-sm text-gray-600 hover:text-gray-900"
                ></button>
            </div>
            <div className="space-y-4">
                {sortedModels.map((model) => (
                    <div
                        key={model._id}
                        className={`p-4 border rounded-lg cursor-pointer transition-colors
                                    ${selectedConfigId === model._id
                                ? 'border-blue-500 bg-blue-50'
                                : 'hover:bg-gray-50'}`}
                        onClick={() => onSelect(model._id)}
                    >
                        <div className="flex justify-between items-start mb-2">
                            <div>
                                <h3 className="font-semibold">{model.name}</h3>
                                <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                            </div>
                            <StatusBadge status={model.status} />
                        </div>

                        <div className="mt-4 space-y-2">
                            <div className="flex items-center gap-2 text-xs text-gray-500">
                                <Clock className="w-4 h-4" />
                                Created: {new Date(model.created_at).toLocaleString()}
                            </div>

                            {model.training_stats && (
                                <div className="space-y-1">
                                    <div className="text-xs text-gray-600">
                                        Records: {model.training_stats.processed_records}
                                    </div>
                                    {model.training_stats.progress !== undefined && (
                                        <Progress
                                            value={model.training_stats.progress}
                                            className="h-1"
                                        />
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// Main Component
export default function ProductSearchConfig() {
    const {
        config,
        setConfig,
        submitConfig,
        isSubmitting,
        error: configError,
        configId,
        validationErrors
    } = useModelConfig();

    const {
        handleFileUpload,
        isUploading,
        csvHeaders,
        error: uploadError,
        uploadProgress
    } = useFileUpload();

    const {
        trainingStatus,
        error: trainingError,
        retryTraining
    } = useTraining(configId);

    const [selectedConfigId, setSelectedConfigId] = useState(null);
    const [csvFile, setCsvFile] = useState(null);
    const [showColumns, setShowColumns] = useState(false);

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

            // Update training config with optimized parameters
            setConfig(prev => ({
                ...prev,
                training_config: {
                    ...prev.training_config,
                    model_type: "transformer",
                    embedding_model: "all-minilm-l6",
                    batch_size: 32,
                    max_tokens: 512,
                    llm_model: "distilgpt2"
                }
            }));

            await submitConfig(csvFile);

        } catch (err) {
            console.error("Submission error:", err);
        }
    };

    // Display consolidated errors
    const hasErrors = configError || uploadError || trainingError;

    return (
        <div className="max-w-7xl mx-auto p-6 space-y-8">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Product Search Configuration</h1>
                    <p className="mt-2 text-gray-600">Configure and train your product search model</p>
                </div>
            </div>

            {/* Error Display */}
            {hasErrors && (
                <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                        {configError || uploadError || trainingError}
                    </AlertDescription>
                </Alert>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Configuration Form */}
                <Card
                    title="Create New Model"
                    subtitle="Configure your product search model"
                >
                    <Input
                        label="Model Name"
                        value={config.name}
                        onChange={(e) => setConfig(prev => ({ ...prev, name: e.target.value }))}
                        placeholder="Enter model name"
                        error={validationErrors.name}
                    />

                    <Input
                        label="Description"
                        value={config.description}
                        onChange={(e) => setConfig(prev => ({ ...prev, description: e.target.value }))}
                        placeholder="Enter description"
                        as="textarea"
                        rows={3}
                    />

                    <Select
                        label="Training Mode"
                        value={config.mode}
                        onChange={(e) => setConfig(prev => ({ ...prev, mode: e.target.value }))}
                        options={[
                            { value: "replace", label: "Replace (New Model)" },
                            { value: "append", label: "Append (Update Existing)" }
                        ]}
                    />

                    {/* File Upload Section */}
                    <FormSection
                        title="Upload Data"
                        description="Upload your product data in CSV format"
                    >
                        <div
                            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors
                                        ${isUploading ? 'border-blue-300 bg-blue-50' : 'border-gray-300 hover:border-blue-500'}`}
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={(e) => {
                                e.preventDefault();
                                const file = e.dataTransfer.files[0];
                                if (file) handleFileUploadWrapper({ target: { files: [file] } });
                            }}
                        >
                            <div className="flex flex-col items-center space-y-4">
                                <Upload className={`h-12 w-12 ${isUploading ? 'text-blue-500' : 'text-gray-400'}`} />
                                <div className="space-y-2">
                                    <h3 className="font-semibold text-gray-900">Upload CSV file</h3>
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
                                    <span className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 
                                                inline-block text-sm font-medium text-gray-700">
                                        {isUploading ? "Uploading..." : "Select File"}
                                    </span>
                                </label>
                            </div>

                            {/* Upload Progress */}
                            {isUploading && (
                                <div className="mt-4">
                                    <Progress value={uploadProgress} className="h-2" />
                                    <p className="mt-2 text-sm text-blue-600">{uploadProgress.toFixed(0)}% uploaded</p>
                                </div>
                            )}
                        </div>

                        {csvFile && (
                            <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4 flex items-center">
                                <CheckCircle2 className="h-4 w-4 text-green-500 mr-2" />
                                <span className="text-sm text-green-700">File loaded: {csvFile.name}</span>
                            </div>
                        )}
                    </FormSection>

                    {/* CSV Configuration */}
                    {csvHeaders.length > 0 && (
                        <FormSection
                            title="CSV Configuration"
                            description="Map your CSV columns to product attributes"
                        >
                            <div
                                className="flex justify-between items-center cursor-pointer mb-4"
                                onClick={() => setShowColumns(!showColumns)}
                            >
                                <h3 className="text-sm font-medium text-gray-700">Available Columns</h3>
                                {showColumns ? (
                                    <ChevronUp className="h-4 w-4" />
                                ) : (
                                    <ChevronDown className="h-4 w-4" />
                                )}
                            </div>

                            {showColumns && (
                                <div className="grid grid-cols-2 gap-2 mb-4">
                                    {csvHeaders.map((header) => (
                                        <div key={header} className="p-2 bg-gray-50 rounded-lg text-sm">
                                            {header}
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Column Mapping */}
                            <div className="space-y-4">
                                {["id", "name", "description", "category"].map((field) => (
                                    <Select
                                        key={field}
                                        label={`${field.charAt(0).toUpperCase() + field.slice(1)} Column`}
                                        value={config.schema_mapping[`${field}_column`]}
                                        onChange={(e) => setConfig(prev => ({
                                            ...prev,
                                            schema_mapping: {
                                                ...prev.schema_mapping,
                                                [`${field}_column`]: e.target.value
                                            }
                                        }))}
                                        options={csvHeaders.map(header => ({
                                            value: header,
                                            label: header
                                        }))}
                                        error={validationErrors[`${field}_column`]}
                                    />
                                ))}
                            </div>
                        </FormSection>
                    )}

                    {/* Model Configuration */}
                    <FormSection
                        title="Model Configuration"
                        description="Configure your search model parameters"
                    >
                        <Select
                            label="LLM Model"
                            value={config.training_config?.llm_model}
                            onChange={(e) => setConfig(prev => ({
                                ...prev,
                                training_config: {
                                    ...prev.training_config,
                                    llm_model: e.target.value
                                }
                            }))}
                            options={Object.entries(AVAILABLE_LLM_MODELS).map(([key, model]) => ({
                                value: key,
                                label: model.name,
                                description: model.description
                            }))}
                        />

                        <div className="mt-4">
                            <h4 className="text-sm font-medium text-gray-700 mb-2">Model Features</h4>
                            <div className="flex flex-wrap gap-2">
                                {AVAILABLE_LLM_MODELS[config.training_config?.llm_model]?.tags.map((tag) => (
                                    <span key={tag} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </FormSection>

                    {/* Submit Button */}
                    <button
                        onClick={handleSubmit}
                        disabled={isSubmitting || !csvFile}
                        className="w-full mt-6 px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                                    disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
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
                </Card>

                {/* Available Models */}
                <div className="space-y-6">
                    <Card>
                        <ModelList
                            onSelect={setSelectedConfigId}
                            selectedConfigId={selectedConfigId}
                        />
                    </Card>
                </div>
            </div>

            {/* Training Status */}
            {configId && trainingStatus && (
                <Card>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <h2 className="text-xl font-semibold text-gray-900">Training Status</h2>
                            <StatusBadge status={trainingStatus.status} />
                        </div>

                        <div className="space-y-4">
                            {/* Progress Information */}
                            {trainingStatus.training_stats && (
                                <div className="space-y-2">
                                    <div className="flex justify-between text-sm text-gray-600">
                                        <span>Processed Records:</span>
                                        <span>{trainingStatus.training_stats.processed_records}</span>
                                    </div>

                                    {trainingStatus.training_stats.training_accuracy && (
                                        <div className="flex justify-between text-sm text-gray-600">
                                            <span>Training Accuracy:</span>
                                            <span>
                                                {(trainingStatus.training_stats.training_accuracy * 100).toFixed(2)}%
                                            </span>
                                        </div>
                                    )}

                                    {trainingStatus.training_stats.progress !== undefined && (
                                        <div>
                                            <div className="flex justify-between text-sm text-gray-600 mb-1">
                                                <span>Overall Progress:</span>
                                                <span>{trainingStatus.training_stats.progress.toFixed(0)}%</span>
                                            </div>
                                            <Progress
                                                value={trainingStatus.training_stats.progress}
                                                className="h-2"
                                            />
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Status Messages */}
                            <div className="text-sm">
                                {trainingStatus.status === ModelStatus.COMPLETED && (
                                    <Alert className="bg-green-50 border-green-200">
                                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                                        <AlertTitle>Training Completed</AlertTitle>
                                        <AlertDescription>
                                            Your model has been successfully trained and is ready to use.
                                        </AlertDescription>
                                    </Alert>
                                )}

                                {trainingStatus.status === ModelStatus.PROCESSING && (
                                    <Alert className="bg-blue-50 border-blue-200">
                                        <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
                                        <AlertTitle>Training in Progress</AlertTitle>
                                        <AlertDescription>
                                            Your model is currently being trained. This may take several minutes.
                                        </AlertDescription>
                                    </Alert>
                                )}

                                {trainingStatus.status === ModelStatus.FAILED && (
                                    <Alert className="bg-red-50 border-red-200">
                                        <XCircle className="h-4 w-4 text-red-500" />
                                        <AlertTitle>Training Failed</AlertTitle>
                                        <AlertDescription>
                                            {trainingStatus.error || "An error occurred during training."}
                                            <button
                                                onClick={retryTraining}
                                                className="mt-2 flex items-center text-sm text-blue-600 hover:text-blue-800"
                                            >
                                                <RefreshCw className="w-4 h-4 mr-1" />
                                                Retry Training
                                            </button>
                                        </AlertDescription>
                                    </Alert>
                                )}

                                {trainingStatus.status === ModelStatus.QUEUED && (
                                    <Alert className="bg-yellow-50 border-yellow-200">
                                        <Clock className="h-4 w-4 text-yellow-500" />
                                        <AlertTitle>In Queue</AlertTitle>
                                        <AlertDescription>
                                            Your training job is queued and will start soon.
                                        </AlertDescription>
                                    </Alert>
                                )}
                            </div>

                            {/* Model Information */}
                            {trainingStatus.status === ModelStatus.COMPLETED && (
                                <div className="mt-4 space-y-2">
                                    <h3 className="text-sm font-medium text-gray-700">Model Information</h3>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="bg-gray-50 p-3 rounded-lg">
                                            <span className="text-xs text-gray-500">Model Type</span>
                                            <p className="text-sm font-medium text-gray-900">
                                                {config.training_config.llm_model}
                                            </p>
                                        </div>
                                        <div className="bg-gray-50 p-3 rounded-lg">
                                            <span className="text-xs text-gray-500">Embedding Model</span>
                                            <p className="text-sm font-medium text-gray-900">
                                                {config.training_config.embedding_model}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Additional Metrics */}
                            {trainingStatus.training_stats?.metrics && (
                                <div className="mt-4">
                                    <h3 className="text-sm font-medium text-gray-700 mb-2">Training Metrics</h3>
                                    <div className="grid grid-cols-3 gap-4">
                                        {Object.entries(trainingStatus.training_stats.metrics).map(([key, value]) => (
                                            <div key={key} className="bg-gray-50 p-3 rounded-lg">
                                                <span className="text-xs text-gray-500">{key}</span>
                                                <p className="text-sm font-medium text-gray-900">
                                                    {typeof value === 'number' ? value.toFixed(4) : value}
                                                </p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </Card>
            )}
        </div>
    );
}