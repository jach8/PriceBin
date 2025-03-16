from typing import Dict, List, Optional, Union, Any, Tuple, ContextManager
import numpy as np
import pandas as pd
import sqlite3 as sql
import datetime as dt
from tqdm import tqdm
import time
import json
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path 
from contextlib import contextmanager
from functools import wraps

from .get_data import UpdateStocks
from .conn_pool import DatabaseConnectionPool as get_pool
from .indicators import Indicators

# Custom exceptions
class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass

class QueryExecutionError(Exception):
    """Raised when SQL query execution fails"""
    pass

class InvalidParameterError(Exception):
    """Raised when invalid parameters are provided"""
    pass

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """Configure logger with file and console handlers"""
    logger = logging.getLogger(name)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger(__name__)

class Prices(UpdateStocks):
    """Class for managing stock price database connections and queries"""
    
    def __init__(self, connections: Dict[str, str]) -> None:
        """
        Initialize price database manager
        
        Args:
            connections: Dictionary containing database connection paths
        
        Raises:
            DatabaseConnectionError: If database connection fails
            FileNotFoundError: If required files are not found
        """
        super().__init__(connections)
        self.execution_start_time = time.time()
        self.pool = get_pool(connections)
        
        try:
            # Validate connection parameters
            required_keys = ['stock_names', 'daily_db', 'intraday_db', 'ticker_path']
            if not all(key in connections for key in required_keys):
                raise InvalidParameterError(f"Missing required connection parameters: {required_keys}")
            
            # Validate file existence
            for key, path in connections.items():
                if not Path(path).exists():
                    raise FileNotFoundError(f"File not found: {path} for {key}")
            
            # Store database mapping
            self.db_mapping = {
                'stock_names': 'stock_names',
                'daily_db': 'daily_db',
                'intraday_db': 'intraday_db'
            }
            
            # Test all database connections
            for db_type in self.db_mapping.keys():
                with self._get_connection(db_type) as conn:
                    conn.execute("SELECT 1")
            
            # Load ticker data
            try:
                with open(connections['ticker_path'], 'r') as f:
                    self.stocks = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ticker JSON file: {e}")
                raise
                
            
            logger.info(f"PriceDB Initialized successfully at {dt.datetime.now()}")
            logger.info("Connection pool initialized")
            
        except (sql.Error, FileNotFoundError, json.JSONDecodeError) as e:
            error_msg = f"Initialization failed: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e
    
    @contextmanager
    def _get_connection(self, db_type: str) -> ContextManager[sql.Connection]:
        """
        Get a database connection from the pool
        
        Args:
            db_type: Type of database ('names', 'daily', or 'intraday')
            
        Returns:
            SQLite connection object from pool
            
        Raises:
            DatabaseConnectionError: If connection fails
        """
        if db_type not in self.db_mapping:
            raise InvalidParameterError(f"Invalid database type: {db_type}")
            
        try:
            with self.pool.get_connection(db_type) as conn:
                yield conn
        except (sql.Error, KeyError) as e:
            error_msg = f"Failed to get {db_type} database connection: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e
            
    def update_stock_prices(self) -> None:
        """Update stock prices in database"""
        logger.info('Starting stock price update')
        try:
            self.update()
            logger.info('Successfully updated stock prices')
        except Exception as e:
            logger.error(f'Failed to update stock prices: {str(e)}')
            raise
            
    def custom_q(self, q: str) -> pd.DataFrame:
        """
        Execute a custom query on the daily_db
        
        Args:
            q: SQL query string
            
        Returns:
            DataFrame containing query results
            
        Raises:
            QueryExecutionError: If query execution fails
        """
        try:
            with self._get_connection('daily') as conn:
                cursor = conn.cursor()
                cursor.execute(q)
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(results, columns=columns)
        except sql.Error as e:
            error_msg = f"Query execution failed: {str(e)}\nQuery: {q}"
            logger.error(error_msg)
            raise QueryExecutionError(error_msg) from e
    
    def _get1minCl(self, stock: str, agg: str = '1min') -> pd.DataFrame:
        """Get 1-minute close prices for a stock"""
        try:
            q = f'''select datetime(date) as date, close from {stock} order by datetime(date) asc'''
            with self._get_connection('intraday') as conn:
                cursor = conn.cursor()
                cursor.execute(q)
                df = pd.DataFrame(cursor.fetchall(), columns=['date', stock])
                df.date = pd.to_datetime(df.date)
                df = df.set_index('date')
                if agg != '1min':
                    df = df.resample(agg).last()
                return df
        except (sql.Error, pd.errors.EmptyDataError) as e:
            logger.error(f"Failed to get 1-minute close for {stock}: {str(e)}")
            raise

    def get_intraday_close(self, stocks: List[str], agg: str = '1min') -> pd.DataFrame:
        """
        Get intraday closing prices for multiple stocks
        
        Args:
            stocks: List of stock symbols
            agg: Aggregation interval
            
        Returns:
            DataFrame with stock prices
            
        Raises:
            InvalidParameterError: If stocks is not a list
        """
        if not isinstance(stocks, list):
            raise InvalidParameterError("Input must be a list of stocks")
            
        try:
            out = [self._get1minCl(stock) for stock in stocks]
            out = [i.resample(agg).last() for i in out]
            return pd.concat(out, axis=1)
        except Exception as e:
            logger.error(f"Failed to get intraday close prices: {str(e)}")
            raise

    def _getClose(self, stock: str) -> pd.DataFrame:
        """Get daily closing prices for a stock"""
        try:
            q = f'''select date(date) as date, close as "Close" from {stock} order by date(date) asc'''
            with self._get_connection('daily') as conn:
                df = pd.read_sql_query(q, conn, parse_dates=['date'], index_col='date')
                return df.rename(columns={'Close': stock})
        except (sql.Error, pd.errors.DatabaseError) as e:
            logger.error(f"Failed to get close prices for {stock}: {str(e)}")
            raise

    def get_close(self, stocks: List[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Get daily closing prices for multiple stocks"""
        if not isinstance(stocks, list):
            raise InvalidParameterError("Input must be a list of stocks")
            
        try:
            out = [self._getClose(stock) for stock in stocks]
            df = pd.concat(out, axis=1)
            
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
                
            return df
        except Exception as e:
            logger.error(f"Failed to get close prices: {str(e)}")
            raise

    def ohlc(self, stock: str, daily: bool = True, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Get OHLCV data for a stock"""
        try:
            if daily:
                q = f'''select date(date) as "Date", open, high, low, close, volume from {stock} order by date(date) asc'''
                db_type = 'daily_db'
            else:
                if start is None:
                    q = f'''select datetime(date) as "Date", open, high, low, close, volume from {stock} order by datetime(date) asc'''
                else:
                    q = f'''
                    select
                        datetime(date) as "Date",
                        open,
                        high,
                        low,
                        close,
                        volume
                    from {stock}
                    where
                        date(date) >= date("{start}")
                    order by datetime(date) asc'''
                db_type = 'intraday_db'
                
            with self._get_connection(db_type) as conn:
                cursor = conn.cursor()
                cursor.execute(q)
                df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
                df.Date = pd.to_datetime(df.Date)
                df.index = df.Date
                
                if start is not None:
                    df = df[df.index >= start]
                if end is not None:
                    df = df[df.index <= end]
                    
                return df
                
        except Exception as e:
            logger.error(f"Failed to get OHLC data for {stock}: {str(e)}")
            raise

    def get_aggregates(self, df: pd.DataFrame, agg_type: str = 'all') -> Dict[str, pd.DataFrame]:
        """
        Get DataFrame aggregations at different time intervals
        
        Args:
            df: DataFrame with datetime index
            agg_type: Type of aggregation to return ('intraday', 'daily', or 'all')
            
        Returns:
            Dictionary of aggregated DataFrames
        
        Raises:
            InvalidParameterError: If DataFrame index is not DatetimeIndex or invalid agg_type
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise InvalidParameterError("DataFrame index must be DatetimeIndex")
            
        if agg_type not in ['intraday', 'daily', 'all']:
            raise InvalidParameterError("agg_type must be 'intraday', 'daily', or 'all'")
            
        try:
            aggregations = {}
            
            if agg_type in ['intraday', 'all']:
                intraday_aggs = {
                    '3min': df.resample('3T').last().dropna(),
                    '6min': df.resample('6T').last().dropna(),
                    '18min': df.resample('18T').last().dropna(),
                    '1H': df.resample('H').last().dropna(),
                    '4H': df.resample('4H').last().dropna()
                }
                aggregations.update(intraday_aggs)
                
            if agg_type in ['daily', 'all']:
                daily_aggs = {
                    'B': df.resample('B').last().dropna(),
                    'W': df.resample('W').last().dropna(),
                    'M': df.resample('M').last().dropna()
                }
                aggregations.update(daily_aggs)
                
            return aggregations
            
        except Exception as e:
            logger.error(f"Failed to compute aggregates: {str(e)}")
            raise

    def daily_aggregates(self, stock: str) -> Dict[str, pd.DataFrame]:
        """Get daily aggregates for a stock"""
        try:
            df = self._getClose(stock)
            return self.get_aggregates(df)
        except Exception as e:
            logger.error(f"Failed to get daily aggregates for {stock}: {str(e)}")
            raise

    def close_connections(self) -> None:
        """Close all pooled connections"""
        try:
            self.pool.close_all()
            
            end_time = time.time()
            runtime_min = (end_time - self.execution_start_time) / 60
            logger.info(f"All connections returned to pool. Total runtime: {runtime_min:.2f} min")
            
        except Exception as e:
            logger.error(f"Error closing pool connections: {str(e)}")
            raise

    def model_preperation(self, stock, daily = True, ma = 'ema', start_date = None, end_date = None) -> Dict:
        """
        Prepare data for model training
        
        Args:
            stock: Stock symbol
            daily: Whether to use daily or intraday data
            
        Returns:
            Tuple of X, y, feature names, and target names
        """
        try:
            i = Indicators()
            df = self.ohlc(stock, daily=daily, start = start_date, end = end_date)
            mdf = i.all_indicators(df, ma).dropna().drop(columns = ['open', 'high', 'low'])
            mdf['target'] = mdf['close'].pct_change().shift(-1)
            mdf = mdf.dropna()
            return {
                'stock': stock,
                'df': mdf,
                'X': mdf.drop(columns = ['close', 'target']),
                'y': mdf['target'],
                'features': list(mdf.drop(columns = ['close', 'target']).columns),
                'target': ['target']
            }

        except Exception as e:
            logger.error(f"Failed to prepare data for model: {str(e)}")
            raise
