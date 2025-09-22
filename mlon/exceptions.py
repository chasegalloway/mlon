"""
Custom exceptions for MLON package.
"""

class MLONError(Exception):
    """Base exception class for MLON."""
    pass

class MLONValueError(MLONError):
    """Raised when a function receives an argument with the right type but inappropriate value."""
    pass

class MLONTypeError(MLONError):
    """Raised when a function receives an argument with an inappropriate type."""
    pass

class MLONRuntimeError(MLONError):
    """Raised when an error occurs during runtime that doesn't fall into other categories."""
    pass

class MLONDataError(MLONError):
    """Raised when there are issues with the input data."""
    pass

class MLONConfigurationError(MLONError):
    """Raised when there are issues with configuration."""
    pass

class MLONIOError(MLONError):
    """Raised when there are issues with I/O operations."""
    pass
