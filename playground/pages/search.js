import React, { useState, useEffect } from 'react';
import {
    Search,
    Loader2,
    Filter,
    ChevronDown,
    CheckCircle2,
    XCircle,
    AlertCircle,
    Clock,
    PlayCircle,
    StopCircle,
    Info,
    ChevronLeft,
    ChevronRight,
    RefreshCw
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

const useModelSearch = () => {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedModel, setSelectedModel] = useState(null);
    const [searchResults, setSearchResults] = useState([]);
    const [searching, setSearching] = useState(false);
    const [activeFilters, setActiveFilters] = useState({});
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/config`);
            if (!response.ok) throw new Error('Failed to fetch models');
            const data = await response.json();
            setModels(data.configs);
        } catch (err) { 
            setError(err.message); 
        } finally {
            setLoading(false);
        }
    };

    const performSearch = async (query, filters = {}, page = 1) => {
        if (!selectedModel || selectedModel.status !== 'completed') return;

        setSearching(true);
        try {
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    config_id: selectedModel._id,
                    max_items: 20,
                    filters: filters,
                    page: page
                })
            });

            if (!response.ok) throw new Error('Search failed');
            const data = await response.json();
            setSearchResults(data.results);
            setTotalPages(data.total_pages || 1);
            setCurrentPage(page);
        } catch (err) {
            setError(err.message);
        } finally {
            setSearching(false);
        }
    };

    const handlePageChange = (newPage) => {
        if (newPage >= 1 && newPage <= totalPages) {
            performSearch(searchQuery, activeFilters, newPage);
        }
    };

    return {
        models,
        loading,
        error,
        selectedModel,
        setSelectedModel,
        searchResults,
        searching,
        performSearch,
        activeFilters,
        setActiveFilters,
        currentPage,
        totalPages,
        handlePageChange,
        setSearching,
        setError ,
        setCurrentPage,
        setTotalPages,
    };
};

const ModelList = ({ models, selectedModel, onSelect }) => {
    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed':
                return <CheckCircle2 className="h-4 w-4 text-green-500" />;
            case 'failed':
                return <XCircle className="h-4 w-4 text-red-500" />;
            case 'pending':
                return <Clock className="h-4 w-4 text-yellow-500" />;
            case 'processing':
                return <PlayCircle className="h-4 w-4 text-blue-500" />;
            case 'canceled':
                return <StopCircle className="h-4 w-4 text-gray-500" />;
            default:
                return <Info className="h-4 w-4 text-gray-500" />;
        }
    };

    return (
        <div className="border dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-800">
            <div className="bg-gray-50 dark:bg-gray-800 px-4 py-2 border-b dark:border-gray-700">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">Available Models</h3>
            </div>
            <div className="divide-y dark:divide-gray-700">
                {models.map(model => (
                    <div
                        key={model._id}
                        onClick={() => onSelect(model)}
                        className={`p-4 cursor-pointer transition-colors
                            ${selectedModel?._id === model._id 
                                ? 'bg-blue-50 dark:bg-blue-900/50' 
                                : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                            } ${model.status !== 'completed' 
                                ? 'opacity-50 cursor-not-allowed' 
                                : ''
                            }`}
                        title={model.status !== 'completed' ? 'Only completed models can be used for searching' : ''}
                    >
                        <div className="flex justify-between items-center">
                            <div>
                                <h4 className="font-medium text-gray-900 dark:text-gray-100">{model.name}</h4>
                                <p className="text-sm text-gray-600 dark:text-gray-400">{model.description}</p>
                            </div>
                            {selectedModel?._id === model._id && (
                                <CheckCircle2 className="h-5 w-5 text-blue-500" />
                            )}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                            Created: {new Date(model.created_at).toLocaleString()}
                        </div>
                        <div className="flex items-center gap-2 mt-2">
                            {getStatusIcon(model.status)}
                            <span className="text-sm text-gray-600">{model.status}</span>
                        </div>
                        <div className="text-xs text-gray-500 mt-2">
                            Version: {model.version}
                        </div>
                        <div className="text-xs text-gray-500 mt-2">
                            Model Path: {model.model_path}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const FilterPanel = ({ model, activeFilters, onFilterChange }) => {
    if (!model?.schema_mapping) return null;

    const { schema_mapping } = model;
    const [categoryFilter, setCategoryFilter] = useState('');
    const [priceRange, setPriceRange] = useState({ min: '', max: '' });
    const [customFilters, setCustomFilters] = useState({});

    const handleFilterApply = () => {
        const filters = {};

        if (categoryFilter) {
            filters.category = categoryFilter;
        }

        if (priceRange.min !== '' || priceRange.max !== '') {
            filters.price = [
                parseFloat(priceRange.min || 0),
                parseFloat(priceRange.max || Number.MAX_SAFE_INTEGER)
            ];
        }

        // Add custom column filters
        Object.entries(customFilters).forEach(([key, value]) => {
            if (value) filters[key] = value;
        });

        onFilterChange(filters);
    };

    return (
        <div className="border dark:border-gray-700 rounded-lg p-4 space-y-4 bg-white dark:bg-gray-800">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <Filter className="h-4 w-4" />
                Filters
            </h3>

            {/* Category Filter */}
            {schema_mapping.category_column && (
                <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Category</label>
                    <input
                        type="text"
                        value={categoryFilter}
                        onChange={e => setCategoryFilter(e.target.value)}
                        placeholder="Filter by category"
                        className="w-full mt-1 p-2 border dark:border-gray-700 rounded bg-white dark:bg-gray-900 
                                 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400"
                    />
                </div>
            )}

            {/* Price Range Filter */}
            <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Price Range</label>
                <div className="flex gap-2 mt-1">
                    <input
                        type="number"
                        value={priceRange.min}
                        onChange={e => setPriceRange(prev => ({ ...prev, min: e.target.value }))}
                        placeholder="Min"
                        className="w-1/2 p-2 border dark:border-gray-700 rounded bg-white dark:bg-gray-900 
                                 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400"
                    />
                    <input
                        type="number"
                        value={priceRange.max}
                        onChange={e => setPriceRange(prev => ({ ...prev, max: e.target.value }))}
                        placeholder="Max"
                        className="w-1/2 p-2 border dark:border-gray-700 rounded bg-white dark:bg-gray-900 
                                 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400"
                    />
                </div>
            </div>

            {/* Custom Column Filters */}
            {schema_mapping.custom_columns?.map(column => (
                <div key={column.standard_column}>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {column.standard_column}
                    </label>
                    <input
                        type="text"
                        value={customFilters[column.standard_column] || ''}
                        onChange={e => setCustomFilters(prev => ({
                            ...prev,
                            [column.standard_column]: e.target.value
                        }))}
                        placeholder={`Filter by ${column.standard_column}`}
                        className="w-full mt-1 p-2 border dark:border-gray-700 rounded bg-white dark:bg-gray-900 
                                 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400"
                    />
                </div>
            ))}

            <button
                onClick={handleFilterApply}
                className="w-full px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded 
                         hover:bg-blue-700 dark:hover:bg-blue-600 transition-colors"
            >
                Apply Filters
            </button>
        </div>
    );
};

const SearchError = ({ error, onRetry }) => (
    <div className="flex flex-col items-center justify-center p-8 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-100 dark:border-red-800">
        <XCircle className="h-12 w-12 text-red-500 mb-4" />
        <h3 className="text-lg font-semibold text-red-700 dark:text-red-400 mb-2">Search Error</h3>
        <p className="text-red-600 dark:text-red-300 text-center mb-4">{error}</p>
        {onRetry && (
            <button
                onClick={onRetry}
                className="px-4 py-2 bg-red-100 dark:bg-red-800 text-red-700 dark:text-red-300 rounded-md hover:bg-red-200 dark:hover:bg-red-700 transition-colors flex items-center gap-2"
            >
                <RefreshCw className="h-4 w-4" />
                Retry Search
            </button>
        )}
    </div>
);

const SearchResults = ({ results, naturalResponse, queryInfo, currentPage, totalPages, onPageChange, error }) => {
    if (error) {
        return <SearchError error={error} />;
    }

    return (
        <div className="space-y-6">
            {/* Natural Language Response */}
            {naturalResponse && (
                <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-100 dark:border-blue-800 
                         rounded-lg p-4 mb-4">
                    <h3 className="text-sm font-semibold text-blue-800 dark:text-blue-300 mb-2">AI Response:</h3>
                    <p className="text-gray-700 dark:text-gray-300">{naturalResponse}</p>
                </div>
            )}

            {/* Query Info */}
            {queryInfo && (
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                    <span>Original Query: {queryInfo.original}</span>
                    <span className="mx-2">•</span>
                    <span>Model: {queryInfo.model_path}</span>
                </div>
            )}

            {/* Results */}
            {results.map((result, index) => (
                <div key={index} className="border dark:border-gray-700 rounded-lg p-4 
                                    hover:shadow-md transition-shadow bg-white dark:bg-gray-800">
                    <div className="flex justify-between items-start">
                        <div>
                            <h3 className="font-semibold text-lg text-gray-900 dark:text-gray-100">{result.name}</h3>
                            <p className="text-gray-600 dark:text-gray-400 mt-1">{result.description}</p>
                        </div>
                        <div className="flex flex-col items-end">
                            <span className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-300 text-xs px-2 py-1 rounded-full">
                                {(result.score * 100).toFixed(1)}% match
                            </span>
                            <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">ID: {result.id}</span>
                        </div>
                    </div>

                    <div className="mt-3 pt-3 border-t dark:border-gray-700">
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <span className="text-sm text-gray-500 dark:text-gray-400">Category:</span>
                                <span className="ml-2 text-sm font-medium text-gray-900 dark:text-gray-100">{result.category}</span>
                            </div>

                            {/* Metadata fields */}
                            {Object.entries(result.metadata || {}).map(([key, value]) => (
                                <div key={key}>
                                    <span className="text-sm text-gray-500 dark:text-gray-400">{key}:</span>
                                    <span className="ml-2 text-sm font-medium text-gray-900 dark:text-gray-100">{value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            ))}

            {/* Pagination */}
            {totalPages > 1 && (
                <div className="flex justify-center items-center gap-4 mt-6">
                    <button
                        onClick={() => onPageChange(currentPage - 1)}
                        disabled={currentPage === 1}
                        className="p-2 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 
                             dark:hover:bg-gray-700 disabled:opacity-50"
                    >
                        <ChevronLeft className="h-4 w-4" />
                    </button>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        Page {currentPage} of {totalPages}
                    </span>
                    <button
                        onClick={() => onPageChange(currentPage + 1)}
                        disabled={currentPage === totalPages}
                        className="p-2 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 
                             dark:hover:bg-gray-700 disabled:opacity-50"
                    >
                        <ChevronRight className="h-4 w-4" />
                    </button>
                </div>
            )}
        </div>
    );
};

export default function ModelSearchComponent() {
    const {
        models,
        loading,
        error,
        selectedModel,
        setSelectedModel,
        searchResults,
        searching,
        performSearch,
        activeFilters,
        setActiveFilters,
        currentPage,
        totalPages,
        setSearching ,
        setError ,
        setCurrentPage,
        handlePageChange,
        setTotalPages,
    } = useModelSearch();

    const [searchQuery, setSearchQuery] = useState('');

    const [searchResponse, setSearchResponse] = useState({
        results: [],
        naturalResponse: '',
        queryInfo: null,
        total: 0
    });

    const [searchError, setSearchError] = useState(null);

    const handleSearch = async (e) => {
        e.preventDefault();
        setSearchError(null); // Reset error state
        setSearching(true);

        try {
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: searchQuery,
                    model_path: selectedModel.model_path,
                    max_items: 20,
                    filters: activeFilters,
                    page: 1
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Search failed');
            }

            // Check if the response contains an error message
            if (data.error) {
                throw new Error(data.error);
            }

            setSearchResponse({
                results: data.results || [],
                naturalResponse: data.natural_response,
                queryInfo: data.query_info,
                total: data.total
            });
            setCurrentPage(1);
            setTotalPages(Math.ceil(data.total / 20));

        } catch (err) {
            setSearchError(err.message);
            setSearchResponse({
                results: [],
                naturalResponse: '',
                queryInfo: null,
                total: 0
            });
        } finally {
            setSearching(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center p-8">
                <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-4 bg-red-50 text-red-700 rounded-lg flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                {error}
            </div>
        );
    }

    return (
        <div className="max-w-7xl mx-auto p-6 bg-white dark:bg-gray-900">
            <h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">Product Search</h1>

            <div className="grid grid-cols-12 gap-6">
                {/* Model Selection */}
                <div className="col-span-12 lg:col-span-3">
                    <ModelList
                        models={models}
                        selectedModel={selectedModel}
                        onSelect={setSelectedModel}
                    />
                </div>

                {/* Search and Results */}
                <div className="col-span-12 lg:col-span-6">
                    <div className="space-y-6">
                        {/* Search Box */}
                        <form onSubmit={handleSearch} className="flex gap-2">
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Enter search query..."
                                className="flex-1 p-2 border dark:border-gray-700 rounded bg-white dark:bg-gray-900 
                                         text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400
                                         focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                disabled={!selectedModel || selectedModel.status !== 'completed'}
                            />
                            <button
                                type="submit"
                                disabled={!selectedModel || !searchQuery || searching || selectedModel.status !== 'completed'}
                                className="px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded 
                                         hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-50 
                                         flex items-center gap-2 transition-colors"
                            >
                                {searching ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                    <Search className="h-4 w-4" />
                                )}
                                Search
                            </button>
                        </form>

                        {/* Error or Results */}
                        {searchError ? (
                            <SearchError 
                                error={searchError} 
                                onRetry={() => handleSearch(new Event('submit'))} 
                            />
                        ) : (
                            searchResponse.results.length > 0 ? (
                                <SearchResults
                                    results={searchResponse.results}
                                    naturalResponse={searchResponse.naturalResponse}
                                    queryInfo={searchResponse.queryInfo}
                                    currentPage={currentPage}
                                    totalPages={totalPages}
                                    onPageChange={handlePageChange}
                                />
                            ) : (
                                <div className="text-center text-gray-500 p-8">
                                    {selectedModel ?
                                        (selectedModel.status === 'completed' ? "Enter a search query to see results" : "Only completed models can be used for searching") :
                                        "Select a model to start searching"
                                    }
                                </div>
                            )
                        )}
                    </div>
                </div>

                {/* Filters */}
                <div className="col-span-12 lg:col-span-3">
                    {selectedModel && selectedModel.status === 'completed' && (
                        <FilterPanel
                            model={selectedModel}
                            activeFilters={activeFilters}
                            onFilterChange={setActiveFilters}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}