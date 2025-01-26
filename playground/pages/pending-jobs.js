import React, { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

const QueuedJobsDisplay = () => {
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [refreshKey, setRefreshKey] = useState(0);

    useEffect(() => {
        const fetchJobs = async () => {
            try {
                setLoading(true);
                const response = await fetch(`${API_BASE_URL}/queue`);
                if (!response.ok) {
                    throw new Error('Failed to fetch jobs');
                }
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
        // Set up auto-refresh every 5 seconds
        const interval = setInterval(() => {
            setRefreshKey(prev => prev + 1);
        }, 5000);

        return () => clearInterval(interval);
    }, [refreshKey]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 animate-spin" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-600">Error: {error}</p>
            </div>
        );
    }

    return (
        <div className="w-full bg-white shadow rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">Queued Jobs ({jobs.length})</h2>
                <button
                    onClick={() => setRefreshKey(prev => prev + 1)}
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                >
                    Refresh
                </button>
            </div>
            <div>
                {jobs.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                        No jobs in queue
                    </div>
                ) : (
                    <div className="space-y-4">
                        {jobs.map((job, index) => (
                            <div
                                key={job.version_id}
                                className="border rounded-lg p-4 hover:bg-gray-50 transition-colors"
                            >
                                <div className="flex justify-between items-start mb-2">
                                    <div className="font-medium">Job #{index + 1}</div>
                                    <div className="text-sm text-gray-500">
                                        Version ID: {job.version_id}
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <h4 className="text-sm font-medium text-gray-500 mb-1">Configuration</h4>
                                        <div className="text-sm space-y-1">
                                            <p>Mode: {job.config.mode || 'replace'}</p>
                                            <p>Batch Size: {job.config.training_config?.batch_size || 128}</p>
                                            <p className="truncate">S3 Path: {job.s3_path}</p>
                                        </div>
                                    </div>

                                    <div>
                                        <h4 className="text-sm font-medium text-gray-500 mb-1">Schema Mapping</h4>
                                        <div className="text-sm space-y-1">
                                            {job.config.schema_mapping && (
                                                <>
                                                    <p>Name: {job.config.schema_mapping.name_column}</p>
                                                    <p>Description: {job.config.schema_mapping.description_column}</p>
                                                    <p>Category: {job.config.schema_mapping.category_column}</p>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <div className="mt-2 text-sm">
                                    <span className="text-gray-500">Data Source: </span>
                                    <span className="font-mono text-xs break-all">
                                        {job.config.data_source?.location}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default QueuedJobsDisplay;