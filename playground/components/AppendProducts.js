import React, { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { PlusCircle, Upload, Loader2, AlertCircle, TabletSmartphone, FileSpreadsheet, Trash2, CheckCircle2, ChevronUp, ChevronDown } from 'lucide-react';
import Papa from 'papaparse';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Tab component for switching between manual and CSV modes
const TabButton = ({ active, onClick, icon: Icon, children }) => (
    <button
        onClick={onClick}
        className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
            active 
                ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-400 font-medium' 
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
        }`}
    >
        <Icon className="w-4 h-4 mr-2" />
        {children}
    </button>
);

const CSVDropzone = ({ onCSVParse, expectedHeaders }) => {
    const onDrop = useCallback(acceptedFiles => {
        const file = acceptedFiles[0];
        if (file) {
            Papa.parse(file, {
                header: true,
                complete: (results) => {
                    const headers = Object.keys(results.data[0]);
                    const missingHeaders = expectedHeaders.filter(h => !headers.includes(h));
                    
                    if (missingHeaders.length > 0) {
                        onCSVParse({ 
                            error: `Missing required columns: ${missingHeaders.join(', ')}`,
                            data: null,
                            file: null
                        });
                        return;
                    }
                    onCSVParse({ data: results.data, error: null, file });
                },
                error: (error) => {
                    onCSVParse({ error: error.message, data: null, file: null });
                }
            });
        }
    }, [expectedHeaders, onCSVParse]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'text/csv': ['.csv']
        },
        maxFiles: 1
    });

    return (
        <div 
            {...getRootProps()} 
            className={`border-2 border-dashed p-8 rounded-lg text-center cursor-pointer transition-colors
                ${isDragActive 
                    ? 'border-blue-500 bg-blue-50 dark:border-blue-400 dark:bg-blue-900/20' 
                    : 'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600'}`}
        >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" />
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                {isDragActive ? 'Drop the CSV file here' : 'Drag & drop a CSV file, or click to select'}
            </p>
        </div>
    );
};

const DataPreview = ({ data, schema, onRemoveRow }) => {
    if (!data || data.length === 0) return null;

    const columns = [
        schema.id_column,
        schema.name_column,
        schema.description_column,
        schema.category_column,
        ...(schema.custom_columns || []).map(col => col.user_column)
    ].filter(Boolean);

    return (
        <div className="mt-6 overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-800">
                    <tr>
                        {columns.map(col => (
                            <th key={col} 
                                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                {col}
                            </th>
                        ))}
                        <th className="px-6 py-3 text-right">Actions</th>
                    </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                    {data.map((row, idx) => (
                        <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                            {columns.map(col => (
                                <td key={col} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                                    {row[col] || '-'}
                                </td>
                            ))}
                            <td className="px-6 py-4 text-right">
                                <button 
                                    onClick={() => onRemoveRow(idx)}
                                    className="text-red-600 dark:text-red-400 hover:text-red-900 dark:hover:text-red-300"
                                >
                                    <Trash2 className="h-4 w-4" />
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

const ModelCard = ({ model, onSelect, isSelected }) => {
    return (
        <div
            onClick={() => onSelect(model)}
            className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                isSelected 
                    ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-900/50 dark:ring-blue-400' 
                    : 'hover:bg-gray-50 dark:hover:bg-gray-800 border-gray-200 dark:border-gray-700'
            } bg-white dark:bg-gray-800`}
        >
            <div className="flex justify-between items-center mb-2">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">{model.name}</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium
                    ${model.status === 'completed' 
                        ? 'bg-green-100 dark:bg-green-900/50 text-green-800 dark:text-green-400' 
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-400'
                    }`}
                >
                    {model.status}
                </span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{model.description}</p>
            <div className="flex flex-wrap gap-2 text-xs">
                <span className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded-full text-gray-600 dark:text-gray-400">
                    {model.training_stats?.processed_records || 0} products
                </span>
                <span className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded-full text-gray-600 dark:text-gray-400">
                    {new Date(model.created_at).toLocaleDateString()}
                </span>
            </div>
        </div>
    );
};

// Add this helper function before the AppendProducts component
const convertProductsToCSV = (products) => {
    if (products.length === 0) return '';
    
    // Get headers from the first product
    const headers = Object.keys(products[0]);
    
    // Create CSV header row
    const csvRows = [headers.join(',')];
    
    // Add data rows
    products.forEach(product => {
        const values = headers.map(header => {
            const value = product[header] || '';
            // Escape values containing commas or quotes
            return value.includes(',') || value.includes('"') 
                ? `"${value.replace(/"/g, '""')}"` 
                : value;
        });
        csvRows.push(values.join(','));
    });
    
    return csvRows.join('\n');
};

// Add this helper function to validate required fields
const validateProducts = (products, schema) => {
    const requiredFields = [schema.id_column, schema.name_column].filter(Boolean);
    return products.every(product => 
        requiredFields.every(field => product[field] && product[field].trim() !== '')
    );
};

// Modify your AppendProducts component
export default function AppendProducts() {
    const [activeTab, setActiveTab] = useState('manual'); // 'manual' or 'csv'
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState(null);
    const [products, setProducts] = useState([{}]);
    const [csvData, setCSVData] = useState(null);
    const [csvFile, setCSVFile] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/config?status=completed`);
            const data = await response.json();
            setModels(data.configs);
        } catch (err) {
            setError('Failed to fetch models');
        }
    };

    const handleCSVParse = ({ data, error: parseError, file }) => {
        if (parseError) {
            setError(parseError);
            return;
        }

        // Validate data against schema
        if (selectedModel && data) {
            const schema = selectedModel.schema_mapping;
            const requiredColumns = [
                schema.id_column,
                schema.name_column
            ].filter(Boolean);

            const headers = Object.keys(data[0] || {});
            const missingColumns = requiredColumns.filter(col => !headers.includes(col));

            if (missingColumns.length > 0) {
                setError(`Missing required columns: ${missingColumns.join(', ')}`);
                return;
            }
        }

        setCSVData(data);
        setCSVFile(file);
        setError('');
    };

    // CSV upload handling
    const onDrop = useCallback((acceptedFiles) => {
        const file = acceptedFiles[0];
        if (file) {
            Papa.parse(file, {
                header: true,
                complete: (results) => {
                    setCSVData(results.data);
                    setCSVFile(file);
                },
                error: (error) => {
                    setError(`Failed to parse CSV: ${error.message}`);
                }
            });
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'text/csv': ['.csv']
        },
        maxFiles: 1
    });

    const handleSubmit = async () => {
        if (!selectedModel) {
            setError('Please select a model first');
            return;
        }

        try {
            setIsSubmitting(true);
            setError('');
            setSuccess('');

            const formData = new FormData();
            let dataToSubmit;
            
            if (activeTab === 'csv') {
                if (!csvFile || !csvData) {
                    throw new Error('Please upload a CSV file');
                }
                formData.append('file', csvFile);
                dataToSubmit = csvData;
            } else {
                // Validate manual entries
                if (!validateProducts(products, selectedModel.schema_mapping)) {
                    throw new Error('Please fill in all required fields for all products');
                }
                
                // Convert products to CSV format
                const csvString = convertProductsToCSV(products);
                const blob = new Blob([csvString], { type: 'text/csv' });
                const file = new File([blob], 'products.csv', { type: 'text/csv' });
                formData.append('file', file);
                dataToSubmit = products;
            }

            // Create config for appending
            const config = {
                name: `Append to ${selectedModel.name}`,
                description: `Appending ${dataToSubmit.length} products`,
                mode: 'append',
                previous_version: selectedModel._id,
                schema_mapping: selectedModel.schema_mapping,
                training_config: selectedModel.training_config,
                data_source: {
                    type: 'csv',
                    file_type: 'csv'
                }
            };

            // Log the request for debugging
            console.log('Submitting config:', config);

            formData.append('config', JSON.stringify(config));

            const response = await fetch(`${API_BASE_URL}/config`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to append products');
            }

            const result = await response.json();
            setSuccess(`Successfully queued ${dataToSubmit.length} products for training. Config ID: ${result.data.config_id}`);
            resetForm();
            
        } catch (err) {
            setError(err.message);
            console.error('Submission error:', err);
        } finally {
            setIsSubmitting(false);
        }
    };

    const resetForm = () => {
        setProducts([{}]);
        setCSVData(null);
        setCSVFile(null);
    };

    const removeRow = (index) => {
        setCSVData(prev => prev.filter((_, idx) => idx !== index));
    };

    const getRequiredHeaders = () => {
        if (!selectedModel) return [];
        const schema = selectedModel.schema_mapping;
        return [
            schema.id_column,
            schema.name_column,
            schema.description_column,
            schema.category_column,
            ...(schema.custom_columns || []).map(col => col.user_column)
        ].filter(Boolean);
    };


    useEffect(() => {
        if (selectedModel) {
            setProducts([{}]);  // Reset products
            setCSVData(null);   // Reset CSV data
            setCSVFile(null);   // Reset CSV file
            setError('');       // Clear any errors
            setSuccess('');     // Clear success message
        }
    }, [selectedModel]); // Only trigger when selectedModel changes

    // Update model selection handler
    const handleModelSelect = (model) => {
        setSelectedModel(model);
        setActiveTab('manual'); // Reset to manual tab when changing models
    };

    return (
        <div className="max-w-7xl mx-auto p-6 space-y-8">
            {/* Title */}
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Append Products</h1>

            {/* Messages */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-center text-red-700 dark:text-red-400">
                    <AlertCircle className="h-4 w-4 mr-2" />
                    <p>{error}</p>
                </div>
            )}
            {success && (
                <div className="bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg p-4 flex items-center text-green-700 dark:text-green-400">
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    <p>{success}</p>
                </div>
            )}

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Model Selection - Left Column */}
                <div className="space-y-4">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Select Model</h2>
                    <div className="space-y-4">
                        {models.map(model => (
                            <ModelCard
                                key={model._id}
                                model={model}
                                onSelect={handleModelSelect} // Use the new handler
                                isSelected={selectedModel?._id === model._id}
                            />
                        ))}
                    </div>
                </div>

                {/* Right Column - Product Addition */}
                <div className="md:col-span-2">
                    {selectedModel ? (
                        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                            {/* Tab Selection */}
                            <div className="flex space-x-4 mb-6">
                                <TabButton
                                    active={activeTab === 'manual'}
                                    onClick={() => setActiveTab('manual')}
                                    icon={TabletSmartphone}
                                >
                                    Manual Entry
                                </TabButton>
                                <TabButton
                                    active={activeTab === 'csv'}
                                    onClick={() => setActiveTab('csv')}
                                    icon={FileSpreadsheet}
                                >
                                    CSV Upload
                                </TabButton>
                            </div>

                            {/* Form Content */}
                            {activeTab === 'manual' ? (
                                <ManualEntryForm
                                    products={products}
                                    setProducts={setProducts}
                                    schema={selectedModel.schema_mapping}
                                />
                            ) : (
                                <CSVUploadForm
                                    getRootProps={getRootProps}
                                    getInputProps={getInputProps}
                                    isDragActive={isDragActive}
                                    csvFile={csvFile}
                                    csvData={csvData}
                                />
                            )}

                            {/* Submit Button */}
                            <button
                                onClick={handleSubmit}
                                disabled={isSubmitting || (activeTab === 'manual' ? products.length === 0 : !csvFile)}
                                className="mt-6 w-full px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg
                                         hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-50 
                                         flex items-center justify-center transition-colors"
                            >
                                {isSubmitting ? (
                                    <>
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                        Processing...
                                    </>
                                ) : (
                                    `Submit ${activeTab === 'manual' ? products.length : csvData?.length || 0} Products`
                                )}
                            </button>
                        </div>
                    ) : (
                        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 text-center text-gray-500 dark:text-gray-400">
                            Please select a model from the left to append products
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// Add these new components at the end of the file
const ManualEntryForm = ({ products, setProducts, schema }) => {
    const [showAdditionalFields, setShowAdditionalFields] = useState({});

    // Get required and optional fields
    const getFields = () => {
        const required = [
            { key: 'id_column', value: schema.id_column },
            { key: 'name_column', value: schema.name_column },
            { key: 'description_column', value: schema.description_column },
            { key: 'category_column', value: schema.category_column }
        ].filter(f => f.value);

        // Get additional fields from schema's custom columns
        const additionalFields = schema.custom_columns?.map(col => ({
            key: col.user_column,
            label: col.standard_column,
            required: col.required,
            role: col.role
        })) || [];

        return { required, additional: additionalFields };
    };

    // Reset additional fields state when schema changes
    useEffect(() => {
        setShowAdditionalFields({});
    }, [schema]);

    // Only render additional fields section if there are actually additional fields
    const fields = getFields();
    const hasAdditionalFields = fields.additional.length > 0;

    return (
        <div className="space-y-4">
            {products.map((product, index) => (
                <div key={index} className="p-4 border rounded-lg">
                    {/* Product Header */}
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="font-medium">Product {index + 1}</h3>
                        {index > 0 && (
                            <button
                                onClick={() => {
                                    const newProducts = [...products];
                                    newProducts.splice(index, 1);
                                    setProducts(newProducts);
                                }}
                                className="text-red-500 hover:text-red-700"
                            >
                                <Trash2 className="h-4 w-4" />
                            </button>
                        )}
                    </div>

                    {/* Required Fields */}
                    <div className="grid grid-cols-2 gap-4 mb-4">
                        {fields.required.map(({ key, value }) => (
                            <div key={key}>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    {key.replace('_column', '').split('_').join(' ').toUpperCase()}
                                    {(key === 'id_column' || key === 'name_column') && ' *'}
                                </label>
                                <input
                                    type="text"
                                    value={product[value] || ''}
                                    onChange={(e) => {
                                        const newProducts = [...products];
                                        newProducts[index] = {
                                            ...newProducts[index],
                                            [value]: e.target.value
                                        };
                                        setProducts(newProducts);
                                    }}
                                    className="w-full p-2 border rounded-md"
                                    required={key === 'id_column' || key === 'name_column'}
                                />
                            </div>
                        ))}
                    </div>

                    {/* Additional Fields Section */}
                    {hasAdditionalFields && (
                        <div className="mt-4 border-t pt-4">
                            <button
                                type="button"
                                onClick={() => setShowAdditionalFields(prev => ({
                                    ...prev,
                                    [index]: !prev[index]
                                }))}
                                className="text-sm text-blue-600 hover:text-blue-800 flex items-center"
                            >
                                {showAdditionalFields[index] ? 
                                    <ChevronUp className="h-4 w-4 mr-1" /> : 
                                    <ChevronDown className="h-4 w-4 mr-1" />
                                }
                                Additional Fields ({fields.additional.length})
                            </button>
                            
                            {showAdditionalFields[index] && (
                                <div className="mt-4 grid grid-cols-2 gap-4">
                                    {fields.additional.map((field) => (
                                        <div key={field.key}>
                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                {field.label}
                                                {field.required && ' *'}
                                            </label>
                                            <input
                                                type="text"
                                                value={product[field.key] || ''}
                                                onChange={(e) => {
                                                    const newProducts = [...products];
                                                    newProducts[index] = {
                                                        ...newProducts[index],
                                                        [field.key]: e.target.value
                                                    };
                                                    setProducts(newProducts);
                                                }}
                                                className="w-full p-2 border rounded-md"
                                                required={field.required}
                                            />
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            ))}

            {/* Add Product Button */}
            <button
                onClick={() => setProducts([...products, {}])}
                className="w-full p-2 border-2 border-dashed rounded-lg text-gray-600 hover:text-gray-900 hover:border-gray-400 flex items-center justify-center"
            >
                <PlusCircle className="w-4 h-4 mr-2" />
                Add Another Product
            </button>
        </div>
    );
};

const CSVUploadForm = ({ getRootProps, getInputProps, isDragActive, csvFile, csvData }) => {
    return (
        <div className="space-y-4">
            <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                    ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
            >
                <input {...getInputProps()} />
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-sm text-gray-600">
                    {isDragActive
                        ? "Drop the CSV file here"
                        : "Drag & drop a CSV file, or click to select"}
                </p>
            </div>

            {csvFile && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <p className="text-sm text-green-700">
                        File loaded: {csvFile.name}
                        {csvData && ` (${csvData.length} products)`}
                    </p>
                </div>
            )}
        </div>
    );
};

