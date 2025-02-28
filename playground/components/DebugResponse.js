import React, { useState } from 'react';

const DebugResponse = ({ data }) => {
    const [isVisible, setIsVisible] = useState(false);

    if (!data) return null;

    return (
        <div className="mt-4 border border-gray-300 dark:border-gray-700 rounded-md">
            <div 
                className="p-2 bg-gray-100 dark:bg-gray-800 cursor-pointer flex justify-between items-center"
                onClick={() => setIsVisible(!isVisible)}
            >
                <h4 className="text-sm font-semibold">Debug Response Structure</h4>
                <span>{isVisible ? '▼' : '▶'}</span>
            </div>
            
            {isVisible && (
                <pre className="p-4 overflow-auto max-h-96 text-xs bg-gray-50 dark:bg-gray-900">
                    {JSON.stringify(data, null, 2)}
                </pre>
            )}
        </div>
    );
};

export default DebugResponse;
