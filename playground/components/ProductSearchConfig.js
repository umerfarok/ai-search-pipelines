import React, { useState, useCallback, useEffect } from "react";
import {
    AlertCircle, Upload, CheckCircle2, XCircle,
    Loader2, Clock, AlertTriangle, ChevronDown, ChevronUp
} from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
const ModelStatus = {
    PENDING: "pending", QUEUED: "queued", PROCESSING: "processing",
    COMPLETED: "completed", FAILED: "failed", CANCELED: "canceled"
};

const AVAILABLE_LLM_MODELS = {
    "gpt2": {
        name: "GPT-2",
        description: "General purpose language model for product search"
    },
    "all-mpnet-base": {
        name: "all-mpnet-base-v2",
        description: "Powerful embedding model for semantic search"
    }
};

// Reusable UI Components
const FormSection = ({ title, children, className = "" }) => (
    <div className={`mt-6 ${className}`}>
        <h3 className="text-sm font-medium text-gray-700 mb-4">{title}</h3>
        {children}
    </div>
);

const Input = ({ label, ...props }) => (
    <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
        <input
            {...props}
            className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
    </div>
);

const Select = ({ label, options, ...props }) => (
    <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
        <select
            {...props}
            className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
            <option value="">Select {label.toLowerCase()}</option>
            {options.map(({ value, label }) => (
                <option key={value} value={value}>{label}</option>
            ))}
        </select>
    </div>
);

const Card = ({ title, children, className = "" }) => (
    <div className={`bg-white border border-gray-200 rounded-lg shadow-sm p-6 ${className}`}>
        {title && <h2 className="text-xl font-semibold text-gray-900 mb-6">{title}</h2>}
        {children}
    </div>
);

// Custom Hooks
const useModelConfig = () => {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState("");
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
            embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
            batch_size: 128,
            max_tokens: 512,
            validation_split: 0.2,
            llm_model: "gpt2",
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

    return { config, setConfig, submitConfig, isSubmitting, error, configId };
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
                    if (lines.length === 0) throw new Error("The CSV file is empty");
                    const headers = lines[0].trim().split(",").map((h) => h.trim());
                    if (headers.length === 0) throw new Error("No headers found in the CSV file");
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
            reader.onerror = () => setError("Failed to read CSV file");
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
        if (configId) monitorTraining(configId);
    }, [configId, monitorTraining]);

    return { trainingStatus, error };
};

// Status Badge Component
const StatusBadge = ({ status }) => {
    const statusConfig = {
        [ModelStatus.COMPLETED]: { color: "green", icon: CheckCircle2 },
        [ModelStatus.FAILED]: { color: "red", icon: XCircle },
        [ModelStatus.PROCESSING]: { color: "blue", icon: Loader2 },
        [ModelStatus.QUEUED]: { color: "yellow", icon: Clock },
        [ModelStatus.PENDING]: { color: "yellow", icon: Clock },
        [ModelStatus.CANCELED]: { color: "gray", icon: AlertTriangle }
    }[status] || { color: "gray", icon: AlertCircle };

    const Icon = statusConfig.icon;

    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-sm font-medium bg-${statusConfig.color}-100 text-${statusConfig.color}-800`}>
            <Icon className="w-4 h-4" />
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

    if (loading) return <div className="p-4 text-gray-600">Loading models...</div>;
    if (error) return <div className="p-4 text-red-500">{error}</div>;

    return (
        <div className="space-y-4">
            {models?.map((model) => (
                <div
                    key={model._id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${selectedConfigId === model._id ? "border-blue-500 bg-blue-50" : "hover:bg-gray-50"
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

// Main Component
export default function ProductSearchConfig() {
    const { config, setConfig, submitConfig, isSubmitting, error: configError, configId } = useModelConfig();
    const { handleFileUpload, isUploading, csvHeaders, error: uploadError } = useFileUpload();
    const { trainingStatus, error: trainingError } = useTraining(configId);
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
            if (!csvFile) throw new Error("Please upload a CSV file first");
            setConfig(prev => ({
                ...prev,
                training_config: {
                    ...prev.training_config,
                    model_type: "transformer",
                    embedding_model: "all-mpnet-base-v2",
                    batch_size: 4,
                    max_tokens: 512,
                }
            }));
            await submitConfig(csvFile);
        } catch (err) {
            console.error("Submission error:", err);
        }
    };

    return (
        <div className="max-w-7xl mx-auto p-6 space-y-8">
            <h1 className="text-3xl font-bold text-gray-900">Product Search Configuration</h1>

            {/* Error Display */}
            {(configError || uploadError || trainingError) && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center text-red-700">
                    <AlertCircle className="h-4 w-4 mr-2" />
                    <p>{configError || uploadError || trainingError}</p>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Configuration Form */}
                <Card title="Create New Model">
                    <Input
                        label="Model Name"
                        value={config.name}
                        onChange={(e) => setConfig(prev => ({ ...prev, name: e.target.value }))}
                        placeholder="Enter model name"
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
                    <FormSection title="Upload Data">
                        <div
                            className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors"
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
                                    <span className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 inline-block text-sm font-medium text-gray-700">
                                        {isUploading ? "Uploading..." : "Select File"}
                                    </span>
                                </label>
                            </div>
                        </div>
                        {csvFile && (
                            <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4 flex items-center">
                                <CheckCircle2 className="h-4 w-4 text-green-500 mr-2" />
                                <span className="text-sm text-green-700">File loaded: {csvFile.name}</span>
                            </div>
                        )}
                    </FormSection>

                    {/* CSV Columns Section */}
                    {csvHeaders.length > 0 && (
                        <FormSection title="CSV Configuration">
                            <div
                                className="flex justify-between items-center cursor-pointer mb-4"
                                onClick={() => setShowColumns(!showColumns)}
                            >
                                <h3 className="text-sm font-medium text-gray-700">Available Columns</h3>
                                {showColumns ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                            </div>
                            {showColumns && (
                                <div className="grid grid-cols-2 gap-2">
                                    {csvHeaders.map((header) => (
                                        <div key={header} className="p-2 bg-gray-50 rounded-lg text-sm">
                                            {header}
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Column Mapping */}
                            <div className="mt-4">
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
                                    />
                                ))}
                            </div>
                        </FormSection>
                    )}

                    {/* Custom Columns Section */}
                    <FormSection title="Custom Columns">
                        <div className="flex justify-between items-center mb-4">
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
                                className="px-3 py-1 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
                            >
                                Add Column
                            </button>
                        </div>
                        {config.schema_mapping.custom_columns.map((col, index) => (
                            <div key={index} className="flex gap-2 items-center mb-2">
                                <Select
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
                                    options={csvHeaders.map(header => ({
                                        value: header,
                                        label: header
                                    }))}
                                />
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
                                    className="flex-1 p-2.5 border border-gray-300 rounded-lg"
                                />
                                <Select
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
                                    options={[
                                        { value: "metadata", label: "Metadata" },
                                        { value: "training", label: "Training" }
                                    ]}
                                />
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
                    </FormSection>

                    {/* Training Configuration */}
                    <FormSection title="Training Configuration">
                        <Select
                            label="LLM Model for Fine-tuning"
                            value={config.training_config?.llm_model || "gpt2"}
                            onChange={(e) => setConfig(prev => ({
                                ...prev,
                                training_config: {
                                    ...prev.training_config,
                                    llm_model: e.target.value
                                }
                            }))}
                            options={Object.entries(AVAILABLE_LLM_MODELS).map(([key, model]) => ({
                                value: key,
                                label: `${model.name} - ${model.description}`
                            }))}
                        />
                    </FormSection>

                    {/* Submit Button */}
                    <button
                        onClick={handleSubmit}
                        disabled={isSubmitting || !csvFile}
                        className="w-full mt-6 px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center"
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
                <Card title="Available Models">
                    <ModelList
                        onSelect={setSelectedConfigId}
                        selectedConfigId={selectedConfigId}
                    />
                </Card>
            </div>

            {/* Training Status */}
            {configId && trainingStatus && (
                <Card>
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-semibold text-gray-900">Training Status</h2>
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
                </Card>
            )}
        </div>
    );
}