import React from 'react';
import PropTypes from 'prop-types';

export const Alert = ({ children, variant }) => {
  const variantClasses = {
    destructive: 'bg-red-50 border border-red-200 text-red-700',
    warning: 'bg-yellow-50 border border-yellow-200 text-yellow-700',
    info: 'bg-blue-50 border border-blue-200 text-blue-700',
    success: 'bg-green-50 border border-green-200 text-green-700',
  };

  return (
    <div className={`p-4 rounded-lg ${variantClasses[variant] || variantClasses.info}`}>
      {children}
    </div>
  );
};

Alert.propTypes = {
  children: PropTypes.node.isRequired,
  variant: PropTypes.oneOf(['destructive', 'warning', 'info', 'success']),
};

export const AlertTitle = ({ children }) => (
  <h3 className="font-semibold mb-2">{children}</h3>
);

AlertTitle.propTypes = {
  children: PropTypes.node.isRequired,
};

export const AlertDescription = ({ children }) => (
  <p>{children}</p>
);

AlertDescription.propTypes = {
  children: PropTypes.node.isRequired,
};