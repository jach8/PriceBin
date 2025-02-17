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

from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Import core components
from .db_connect import (
    Prices,
    DatabaseConnectionError,
    QueryExecutionError,
    InvalidParameterError
)
from .indicators import Indicators
from src.get_data import UpdateStocks, StockDataError
from src.report import perf as Performance
from src.utils import add_stock, delete_stock

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

def create_default_config(database_root: Optional[Path] = None) -> Dict[str, str]:
    """
    Create default configuration dictionary with database paths.

    Args:
        database_root: Optional root directory for database files.
                      If not provided, uses 'data' in current directory.

    Returns:
        Dictionary containing default database connection paths
    """
    if database_root is None:
        database_root = Path('data')

    return {
        'daily_db': str(database_root / 'prices/stocks.db'),
        'intraday_db': str(database_root / 'prices/stocks_intraday.db'),
        'ticker_path': str(database_root / 'stocks/tickers.json'),
        'stock_names': str(database_root / 'stocks/stock_names.db')
    }
    
    
def Initialize(connections_path: str = 'connections.json') -> None:
    """Create the database connections, stock names and initialize the program.
    
    This function sets up the necessary file structure and databases for the application.
    It creates required directories, initializes database files, and optionally allows
    adding initial stocks to the database.

    Raises:
        IOError: If there's an error creating directories or files.
        sqlite3.Error: If there's an error initializing the database.
    """
    logger.info("Starting initialization")
    
    # Load database configurations from connections.json
    logger.debug("Loading database configurations")
    connections = json.load(open(connections_path, 'r'))
    
    for key, value in connections.items():
        if '.' in value:
            # Handle file paths
            folder = value.split('/')[:-1]
            folder = '/'.join(folder)
            
            if not os.path.exists(folder):
                logger.info(f"Creating directory: {folder}")
                os.makedirs(folder)
                
            if 'db' in value:
                logger.info(f"Initializing database: {value}")
                conn = sql.connect(value)
                conn.close()
            elif 'json' in value:
                logger.info(f"Creating JSON file: {value}")
                with open(value, 'w') as f:
                    if 'ticker' in value:
                        json.dump({'all_stocks': []}, f)
                    else:
                        json.dump({}, f)
            elif 'pkl' in value:
                logger.info(f"Creating pickle file: {value}")
                pickle.dump({}, open(value, 'wb'))
        else:
            # Handle directory paths
            if not os.path.exists(value):
                logger.info(f"Creating directory: {value}")
                os.makedirs(value)
                
    logger.info("File initialization complete")
    
    # Add stocks to database if requested
    print("Initialized Files, Would you like to add stocks to the database? (y/n)")
    add = input()
    
    if add == 'y':
        conn = sql.connect(connections['stock_names'])
        stock = input('Enter the stock you would like to add, if adding more than one stock seperate them with a comma: ')
        
        if ',' in stock:
            stock = stock.split(',')
            for s in stock:
                add_stock(conn=conn, path=connections['ticker_path'], stock=s)
        else:
            add_stock(conn=conn, path=connections['ticker_path'], stock=stock)
    else:
        logger.info("No stocks added during initialization")