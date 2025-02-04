import React, { useState, useEffect } from 'react';
import { Loader2, RefreshCw, Clock, AlertCircle, CheckCircle2, XCircle } from 'lucide-react';
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

const JobCard = ({ job, index }) => (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-all duration-200">
        {/* Header Section */}
        <div className="flex justify-between items-start mb-4">
            <div>
                <h3 className="font-medium text-gray-900 dark:text-gray-100">
                    {job.config.name || `Job #${index + 1}`}
                </h3>
                <div className="mt-1 space-y-1">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                        Config ID: {job.config_id}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                        Version: {job.config.version}
                    </p>
                </div>
            </div>
            <JobStatus status={job.config.status} />
        </div>

        {/* Details Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            {/* Configuration Details */}
            <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Configuration</h4>
                <div className="text-sm space-y-1.5 text-gray-600 dark:text-gray-400">
                    <p>Mode: {job.config.mode}</p>
                    <p>Model: {job.config.training_config.embedding_model.split('/').pop()}</p>
                    <p>Batch Size: {job.config.training_config.batch_size}</p>
                    <p className="truncate">Path: {job.config.model_path}</p>
                    {job.config.previous_version && (
                        <p className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 px-2 py-1 rounded inline-block">
                            Previous Version: {job.config.previous_version}
                        </p>
                    )}
                </div>
            </div>

            {/* Schema Details */}
            <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Schema Mapping</h4>
                <div className="text-sm space-y-1.5 text-gray-600 dark:text-gray-400">
                    {job.config.schema_mapping && (
                        <>
                            <p>ID: {job.config.schema_mapping.id_column}</p>
                            <p>Name: {job.config.schema_mapping.name_column}</p>
                            <p>Description: {job.config.schema_mapping.description_column}</p>
                            <p>Category: {job.config.schema_mapping.category_column}</p>
                            {job.config.schema_mapping.custom_columns?.length > 0 && (
                                <p>Custom Columns: {job.config.schema_mapping.custom_columns.length}</p>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>

        {/* Training Stats */}
        {job.config.training_stats && (
            <div className="mt-4 space-y-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg p-3">
                <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Started:</span>
                    <span className="text-gray-900 dark:text-gray-300">
                        {new Date(job.config.training_stats.start_time).toLocaleString()}
                    </span>
                </div>
                
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Progress:</span>
                        <span className="text-gray-900 dark:text-gray-300">
                            {job.config.training_stats.processed_records} / {job.config.training_stats.total_records || '?'}
                        </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                            className="bg-blue-600 dark:bg-blue-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${job.config.training_stats.progress || 0}%` }}
                        />
                    </div>
                </div>
            </div>
        )}

        {/* Timestamps */}
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>Created: {new Date(job.config.created_at).toLocaleString()}</span>
            <span>Updated: {new Date(job.config.updated_at).toLocaleString()}</span>
        </div>
    </div>
);

// Update JobStatus component to handle all possible states
const JobStatus = ({ status }) => {
    const statusConfig = {
        pending: {
            color: "text-yellow-700 dark:text-yellow-400",
            bgColor: "bg-yellow-100 dark:bg-yellow-900/30",
            icon: Clock,
            label: "Pending"
        },
        queued: {
            color: "text-blue-700 dark:text-blue-400",
            bgColor: "bg-blue-100 dark:bg-blue-900/30",
            icon: Clock,
            label: "Queued"
        },
        processing: {
            color: "text-blue-700 dark:text-blue-400",
            bgColor: "bg-blue-100 dark:bg-blue-900/30",
            icon: Loader2,
            label: "Processing",
            animate: true
        },
        completed: {
            color: "text-green-700 dark:text-green-400",
            bgColor: "bg-green-100 dark:bg-green-900/30",
            icon: CheckCircle2,
            label: "Completed"
        },
        failed: {
            color: "text-red-700 dark:text-red-400",
            bgColor: "bg-red-100 dark:bg-red-900/30",
            icon: XCircle,
            label: "Failed"
        }
    }[status?.toLowerCase()] || {
        color: "text-gray-700 dark:text-gray-400",
        bgColor: "bg-gray-100 dark:bg-gray-800",
        icon: AlertCircle,
        label: status || "Unknown"
    };

    const Icon = statusConfig.icon;

    return (
        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium
            ${statusConfig.color} ${statusConfig.bgColor}`}>
            <Icon className={`w-3.5 h-3.5 mr-1.5 ${statusConfig.animate ? 'animate-spin' : ''}`} />
            {statusConfig.label}
        </span>
    );
};

export default function QueuedJobsDisplay() {
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [refreshKey, setRefreshKey] = useState(0);

    useEffect(() => {
        const fetchJobs = async () => {
            try {
                setLoading(true);
                const response = await fetch(`${API_BASE_URL}/queue`);
                if (!response.ok) throw new Error('Failed to fetch jobs');
                const data = await response.json();
                setJobs(data.jobs);
                setError(null);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchJobs();
        const interval = setInterval(() => setRefreshKey(prev => prev + 1), 5000);
        return () => clearInterval(interval);
    }, [refreshKey]);

    return (
        <div className="max-w-7xl mx-auto p-6 bg-white dark:bg-gray-900 min-h-screen">
            {/* Header Section */}
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                    Processing Queue
                </h1>
                <button
                    onClick={() => setRefreshKey(prev => prev + 1)}
                    className="inline-flex items-center px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white 
                             rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 transition-colors"
                >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                </button>
            </div>

            {/* Loading State */}
            {loading ? (
                <div className="flex items-center justify-center h-64">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-600 dark:text-blue-400" />
                </div>
            ) : error ? (
                <Alert variant="destructive" className="bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription className="text-red-800 dark:text-red-200">{error}</AlertDescription>
                </Alert>
            ) : jobs.length === 0 ? (
                <div className="text-center py-12">
                    <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-6 max-w-md mx-auto">
                        <Clock className="w-12 h-12 text-gray-400 dark:text-gray-500 mx-auto mb-4" />
                        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                            No Active Jobs
                        </h3>
                        <p className="mt-2 text-gray-600 dark:text-gray-400">
                            The processing queue is currently empty.
                        </p>
                    </div>
                </div>
            ) : (
                <div className="grid gap-4 md:gap-6">
                    {jobs.map((job, index) => (
                        <JobCard key={job.config_id} job={job} index={index} />
                    ))}
                </div>
            )}
        </div>
    );
}