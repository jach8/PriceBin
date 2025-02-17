# PriceBin

A comprehensive Python package for stock price tracking, analysis, and database management with support for technical indicators and performance reporting.

## Table of Contents
- [PriceBin](#pricebin)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage Guide](#usage-guide)
    - [Initialize and Add Stocks](#initialize-and-add-stocks)
    - [Fetch and Analyze Price Data](#fetch-and-analyze-price-data)
    - [Generate Performance Reports](#generate-performance-reports)
  - [API Reference](#api-reference)
    - [Core Classes](#core-classes)
      - [Prices](#prices)
      - [Indicators](#indicators)
    - [Utility Functions](#utility-functions)
      - [add\_stock](#add_stock)
      - [delete\_stock](#delete_stock)
  - [Testing](#testing)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Logging](#logging)
    - [Database Maintenance](#database-maintenance)

## Overview

PriceBin is a financial data management system that provides:
- Automated stock price data collection and storage
- Technical analysis tools and indicators
- Performance tracking and reporting
- Database management with SQLite
- Stock portfolio management utilities

## Directory Structure

```
PriceBin/
├── Initialize.py           # Database initialization script
├── connections.json       # Database configuration
├── setup.py              # Package setup script
├── src/
│   ├── __init__.py       # Package initialization
│   ├── db_connect.py     # Database connection management
│   ├── get_data.py       # Data retrieval from Yahoo Finance
│   ├── indicators.py     # Technical analysis tools
│   ├── report.py         # Performance reporting
│   |── utils.py          # Utility functions
│   ├── data/            # Database storage
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PriceBin
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python Initialize.py
```

## Configuration

1. Create `connections.json` with your database paths:

```json
{
    "daily_db": "src/data/stocks.db",
    "intraday_db": "src/data/stocks_intraday.db",
    "ticker_path": "src/data/stocks/tickers.json",
    "stock_names": "src/data/stocks/stock_names.db"
}
```

2. The initialization script will:
   - Create necessary directories
   - Initialize database files
   - Set up tracking configuration
   - Prompt for initial stock symbols

## Usage Guide

### Initialize and Add Stocks

```python
# Initialize database and add stocks
from PriceBin.Initialize import Initialize
Initialize()  # Will prompt for stocks to add

# Add stocks manually
from PriceBin.src.utils.add_stocks import add_stock
import sqlite3

conn = sqlite3.connect('src/data/stocks/stock_names.db')
add_stock(conn, 'src/data/stocks/tickers.json', 'AAPL')
```

### Fetch and Analyze Price Data

```python
from PriceBin import Prices, Indicators

# Initialize price manager
connections = {
    'daily_db': 'src/data/stocks.db',
    'intraday_db': 'src/data/stocks_intraday.db',
    'ticker_path': 'src/data/stocks/tickers.json',
    'stock_names': 'src/data/stocks/stock_names.db'
}

price_db = Prices(connections)

# Get OHLCV data
data = price_db.ohlc('AAPL', daily=True)

# Calculate technical indicators
ind = Indicators(data)
analysis = ind.indicator_df(
    fast=10,    # Fast period
    medium=20,  # Medium period
    slow=50     # Slow period
)
```

### Generate Performance Reports

```python
from PriceBin import Performance

perf = Performance(connections)
returns = perf.get_returns()
perf.show_performance(N=10)  # Show top 10 performers
```

## API Reference

### Core Classes

#### Prices
```python
class Prices:
    """Core class for database interactions and price data retrieval"""
    
    def __init__(self, connections: Dict[str, str]):
        """
        Initialize price database manager
        
        Args:
            connections: Database configuration dictionary
        """
        
    def ohlc(self, stock: str, daily: bool = True) -> pd.DataFrame:
        """
        Get OHLCV data for a stock
        
        Args:
            stock: Stock symbol
            daily: If True, get daily data; if False, get intraday
            
        Returns:
            DataFrame with OHLCV data
        """
```

#### Indicators
```python
class Indicators:
    """Technical analysis calculations"""
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize with price data
        
        Args:
            price_data: OHLCV DataFrame
        """
    
    def indicator_df(self, fast: int = 10, medium: int = 14,
                    slow: int = 35, m: float = 2) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            fast: Fast period window
            medium: Medium period window
            slow: Slow period window
            m: Multiplier for bands
            
        Returns:
            DataFrame with technical indicators
        """
```

### Utility Functions

#### add_stock
```python
def add_stock(conn: sqlite3.Connection, path: str, stock: str) -> None:
    """
    Add a stock to tracking database
    
    Args:
        conn: Database connection
        path: Path to ticker JSON file
        stock: Stock symbol to add
    """
```

#### delete_stock
```python
def delete_stock(conn: sqlite3.Connection, stock: str) -> None:
    """
    Remove a stock from tracking
    
    Args:
        conn: Database connection
        stock: Stock symbol to remove
    """
```

## Testing

1. Install test dependencies:
```bash
pip install pytest pytest-cov
```

2. Run tests:
```bash
pytest tests/
```

## Troubleshooting

### Common Issues

1. **Initialization Errors**
   - Ensure all paths in connections.json are valid
   - Check directory permissions
   - Verify SQLite is installed and working

2. **Data Fetching Issues**
   - Check internet connection
   - Verify Yahoo Finance API access
   - Ensure stock symbols are valid

3. **Database Errors**
   ```python
   from PriceBin import DatabaseConnectionError, QueryExecutionError
   
   try:
       price_db = Prices(connections)
   except DatabaseConnectionError as e:
       print(f"Connection failed: {e}")
   ```

### Logging

Enable debug logging for troubleshooting:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)
```

### Database Maintenance

- Regularly backup database files
- Monitor disk space in data directory
- Use `VACUUM` command periodically on SQLite databases

For more help, check the [issues page](https://github.com/yourusername/PriceBin/issues) or contact support.