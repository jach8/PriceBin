import sqlite3
import logging
from typing import Dict, Optional
from contextlib import contextmanager
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """A thread-safe connection pool for SQLite databases.

    This class manages a pool of SQLite database connections based on configuration
    from a dictionary. It provides connection pooling and proper error handling.

    Attributes:
        _pools: Dictionary mapping connection names to their respective connections.
        _locks: Dictionary of thread locks for each connection.
        _config: Dictionary containing database connection configurations.
    """

    def __init__(self, config: Dict[str, str]) -> None:
        """Initialize the connection pool.

        Args:
            config: Dictionary containing database paths.

        Raises:
            ValueError: If the configuration is invalid.
        """
        self._pools: Dict[str, Optional[sqlite3.Connection]] = {}
        self._locks: Dict[str, Lock] = {}
        self._config: Dict[str, str] = config
        
        logger.info("Initializing database connection pool")
        if not isinstance(config, dict):
            logger.error("Invalid configuration: expected a dictionary")
            raise ValueError("Configuration must be a dictionary")

        # Initialize connection pools for each database
        for db_name, db_path in self._config.items():
            if '.db' in db_path:
                logger.debug(f"Setting up pool for database: {db_name}")
                self._pools[db_name] = None
                self._locks[db_name] = Lock()

    def _create_connection(self, db_name: str) -> sqlite3.Connection:
        """Create a new database connection.

        Args:
            db_name: Name of the database from the configuration.

        Returns:
            A new SQLite connection.

        Raises:
            KeyError: If the database name is not in the configuration.
            sqlite3.Error: If the connection cannot be established.
        """
        if db_name not in self._config:
            logger.error(f"Unknown database name: {db_name}")
            raise KeyError(f"Database '{db_name}' not found in configuration")

        db_path = self._config[db_name]
        logger.debug(f"Creating new connection to {db_path}")
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable row factory for better data access
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database {db_path}: {e}")
            raise

    @contextmanager
    def get_connection(self, db_name: str) -> sqlite3.Connection:
        """Get a database connection from the pool.

        This is a context manager that provides thread-safe access to database
        connections and ensures proper connection handling.

        Args:
            db_name: Name of the database from the configuration.

        Yields:
            A SQLite connection from the pool.

        Raises:
            KeyError: If the database name is not in the configuration.
            sqlite3.Error: If there's an error with the database connection.
        """
        if db_name not in self._locks:
            logger.error(f"Attempted to access non-existent database: {db_name}")
            raise KeyError(f"No connection pool for database '{db_name}'")

        with self._locks[db_name]:
            if self._pools[db_name] is None:
                logger.debug(f"Creating new connection for {db_name}")
                self._pools[db_name] = self._create_connection(db_name)

            try:
                yield self._pools[db_name]
            except sqlite3.Error as e:
                logger.error(f"Database error for {db_name}: {e}")
                # Close and remove the failed connection
                if self._pools[db_name]:
                    try:
                        self._pools[db_name].close()
                    except:
                        pass
                self._pools[db_name] = None
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {db_name}: {e}")
                raise

    def close_all(self) -> None:
        """Close all database connections in the pool.

        This method should be called when shutting down the application to
        ensure all database connections are properly closed.
        """
        logger.info("Closing all database connections")
        for db_name, conn in self._pools.items():
            if conn is not None:
                try:
                    with self._locks[db_name]:
                        conn.close()
                        self._pools[db_name] = None
                        logger.debug(f"Closed connection to {db_name}")
                except sqlite3.Error as e:
                    logger.error(f"Error closing connection to {db_name}: {e}")

# Example usage:
"""
config = {
    'stock_names': '/path/to/stock_names.db',
    'another_db': '/path/to/another.db'
}

pool = DatabaseConnectionPool(config)

try:
    # Get a connection from the pool
    with pool.get_connection('stock_names') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM stocks')
        rows = cursor.fetchall()
        
except sqlite3.Error as e:
    logger.error(f"Database error: {e}")
except KeyError as e:
    logger.error(f"Configuration error: {e}")
finally:
    # Close all connections when done
    pool.close_all()
"""
