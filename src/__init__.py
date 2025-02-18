"""
PriceBin - A Python package for stock price tracking and analysis.

This package provides tools for:
- Stock price data management and retrieval
- Technical analysis with various indicators
- Performance reporting and analysis
- Database management and synchronization

Main Components:
- Prices: Core class for database interactions and price data retrieval
- Indicators: Technical analysis calculations
- UpdateStocks: Stock data fetching and updating
- Performance: Stock performance reporting and analysis
"""

# Import core components
from .db_connect import (
    Prices,
    DatabaseConnectionError,
    QueryExecutionError,
    InvalidParameterError
)
from .indicators import Indicators
from .get_data import UpdateStocks, StockDataError
from .report import perf as Performance
from .utils import add_stock, delete_stock

# Define package version
__version__ = '1.0.0'

# Define public interface
__all__ = [
    # Core classes
    'Prices',
    'Indicators',
    'UpdateStocks',
    'Performance',
    
    # Exceptions
    'DatabaseConnectionError',
    'QueryExecutionError',
    'InvalidParameterError',
    'StockDataError',
]

