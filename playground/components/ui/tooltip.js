// components/ui/Tooltip.js
import React, { useState } from 'react';

export const TooltipProvider = ({ children }) => {
    return <div className="relative">{children}</div>;
};

export const TooltipTrigger = ({ children, onMouseEnter, onMouseLeave }) => {
    return (
        <div
            onMouseEnter={onMouseEnter}
            onMouseLeave={onMouseLeave}
            className="inline-block"
        >
            {children}
        </div>
    );
};

export const TooltipContent = ({ children, visible }) => {
    return (
        visible && (
            <div className="absolute bg-gray-700 text-white text-sm rounded py-1 px-2 z-10">
                {children}
            </div>
        )
    );
};

export const Tooltip = ({ children, content }) => {
    const [visible, setVisible] = useState(false);

    return (
        <TooltipProvider>
            <TooltipTrigger
                onMouseEnter={() => setVisible(true)}
                onMouseLeave={() => setVisible(false)}
            >
                {children}
            </TooltipTrigger>
            <TooltipContent visible={visible}>{content}</TooltipContent>
        </TooltipProvider>
    );
};