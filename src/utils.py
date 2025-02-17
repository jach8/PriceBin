import pandas as pd
import datetime as dt
import sqlite3 as sql
import json
import os
import pickle
import logging
from typing import Optional, List, Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: str) -> Dict:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file to read.

    Returns:
        Dict containing the parsed JSON data.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    logger.debug(f"Loading JSON file from {path}")
    with open(path, 'r') as f:
        return json.load(f)


def update_json(conn: sql.Connection, path: str) -> None:
    """Update the stocks list in a JSON file based on database content.

    Args:
        conn: SQLite database connection.
        path: Path to the JSON file to update.

    Raises:
        sqlite3.Error: If there's an error reading from the database.
        IOError: If there's an error writing to the JSON file.
    """
    logger.debug(f"Updating stocks list in JSON file {path}")
    stock_df = pd.read_sql('SELECT * FROM stocks', conn)
    stock_list = stock_df['stock'].to_list()
    j = load_json(path)
    j['all_stocks'] = stock_list
    
    logger.debug(f"Writing updated stock list with {len(stock_list)} stocks")
    with open(path, 'w') as f:
        json.dump(j, f)


def check_if_stock_exists(conn: sql.Connection, d: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Check if stocks table exists and handle stock data accordingly.

    Args:
        conn: SQLite database connection.
        d: Optional DataFrame containing new stock data to add.

    Returns:
        pd.DataFrame containing either existing stocks or newly added stocks.

    Raises:
        ValueError: If no stock data is provided when table doesn't exist.
        sqlite3.Error: If there's an error accessing the database.
    """
    logger.debug("Checking stocks table existence")
    tables = pd.read_sql('SELECT name FROM sqlite_master WHERE type = "table"', conn)
    
    if 'stocks' not in tables['name'].to_list():
        if d is None:
            logger.error("No stock data provided for non-existent table")
            raise ValueError('No stock data to enter in the database')
        else:
            logger.info("Creating new stocks table")
            d.to_sql('stocks', conn, index=False)
            return d
    else:
        stock_df = pd.read_sql('SELECT * FROM stocks', conn)
        stock_list = stock_df['stock'].to_list()
        if d is None:
            logger.debug("Returning existing stock data")
            return stock_df
        else:
            logger.info(f"Appending new stocks to existing table")
            d = d[~d['stock'].isin(stock_list)]
            d.to_sql('stocks', conn, index=False, if_exists='append')
            return d


def add_stock(conn: sql.Connection, path: str, stock: str) -> None:
    """Add a stock to the database and update the tickers file.

    Args:
        conn: Connection to the stock_names database.
        path: Path to the ticker json file.
        stock: Stock symbol to add to the database.

    Raises:
        sqlite3.Error: If there's an error writing to the database.
        IOError: If there's an error updating the JSON file.
    """
    logger.info(f"Adding stock {stock} to database")
    stock_less_special_chars = stock.replace('^', '')
    d = pd.DataFrame({
        'date': [dt.datetime.today().strftime('%Y-%m-%d')],
        'stock': [stock]
    })
    
    stock_df = check_if_stock_exists(conn, d)
    conn.commit()
    update_json(conn, path)
    logger.info(f"Stock {stock} successfully added to database")


def delete_stock(conn: sql.Connection, path: str, stock: Optional[str] = None) -> None:
    """Delete a stock from the database and update the tickers file.

    Args:
        conn: Database connection.
        path: Path to the tickers JSON file.
        stock: Stock symbol to delete.

    Raises:
        sqlite3.Error: If there's an error modifying the database.
        IOError: If there's an error updating the JSON file.
    """
    logger.info(f"Deleting stock {stock} from database")
    stock_df = pd.read_sql('SELECT * FROM stocks', conn)
    out = stock_df[stock_df.stocks != stock]
    out.to_sql('stocks', conn, if_exists='replace', index=False)
    update_json(conn, path)
    logger.info(f"Stock {stock} successfully deleted from database")

