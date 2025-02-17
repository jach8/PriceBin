from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from ..indicators import Indicators
from ..backtest.PortfolioStats import PortfolioStats
from ..backtest.simulator import MarketSim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Abstract base class for implementing trading strategies.
    
    This class provides the framework for creating testable trading strategies
    that integrate with the existing backtesting infrastructure. It handles:
    - Strategy initialization and configuration
    - Technical indicator calculation
    - Signal generation
    - Performance evaluation
    - Position tracking

    Attributes:
        name: Strategy name
        params: Strategy parameters dictionary
        market_sim: Market simulator instance
        indicators: Technical indicator calculator
        positions: List of position dictionaries
    """

    def __init__(self, name: str, market_sim: MarketSim, params: Optional[Dict] = None):
        """Initialize strategy.

        Args:
            name: Strategy identifier
            market_sim: Initialized MarketSim instance for market operations
            params: Strategy configuration parameters
        """
        self.name = name
        self.market_sim = market_sim
        self.params = params or {}
        self.indicators: Optional[Indicators] = None
        self.positions: List[Dict] = []
        self._performance_stats: Optional[Dict] = None
        
        logger.info(f"Initialized strategy: {name}")
        logger.debug(f"Strategy parameters: {params}")

    def prepare_data(self, data: pd.DataFrame) -> None:
        """Prepare price data and initialize indicators.

        Args:
            data: OHLCV price data

        Raises:
            ValueError: If data is missing required columns
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col.lower() in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
            
        try:
            self.data = data.copy()
            self.indicators = Indicators(self.data)
            logger.info("Data prepared and indicators initialized")
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """Generate trading signals from indicator data.
        
        This method must be implemented by strategy subclasses to define the
        strategy's trading logic and signal generation rules.
        
        Returns:
            pd.Series of trading signals where:
                1 = Long entry
                -1 = Short entry
                0 = Exit/no position
        """
        pass

    def convert_signals_to_orders(self, signals: pd.Series, trade_size: int = 100) -> pd.DataFrame:
        """Convert strategy signals to order format compatible with MarketSim.

        Args:
            signals: Trading signals series (-1, 0, 1)
            trade_size: Number of shares per trade

        Returns:
            DataFrame with columns [Symbol, Order, Shares] formatted for MarketSim
        """
        try:
            orders = pd.DataFrame(index=signals.index)
            orders['Symbol'] = self.data.columns[0].upper()  # Assume first column is price data
            orders['Shares'] = trade_size
            
            # Convert signals to BUY/SELL/HOLD orders
            orders['Order'] = 'HOLD'
            orders.loc[signals == 1, 'Order'] = 'BUY'
            orders.loc[signals == -1, 'Order'] = 'SELL'
            
            logger.debug(f"Converted {len(signals)} signals to orders")
            return orders
        
        except Exception as e:
            logger.error(f"Error converting signals to orders: {str(e)}")
            raise

    def backtest(self, data: pd.DataFrame, trade_size: int = 100, 
                initial_capital: float = 100000.0,
                commission: float = 9.95, impact: float = 0.005) -> Tuple[pd.DataFrame, Dict]:
        """Run backtest of strategy on historical data.

        Args:
            data: OHLCV price data
            trade_size: Number of shares per trade
            initial_capital: Starting capital
            commission: Commission per trade
            impact: Market impact cost

        Returns:
            Tuple containing:
                - DataFrame of trades and portfolio values
                - Dict of performance metrics

        Raises:
            ValueError: If data validation fails
        """
        try:
            # Prepare data and generate signals
            self.prepare_data(data)
            signals = self.generate_signals()
            
            # Convert signals to orders
            orders = self.convert_signals_to_orders(signals, trade_size)
            
            # Execute orders through market simulator
            results = self.market_sim.compute_portvals(
                orders=orders,
                startval=initial_capital,
                commission=commission,
                impact=impact
            )
            
            # Calculate performance metrics
            portfolio_stats = PortfolioStats(results['portfolio']['port_val'], 0.0)
            self._performance_stats = portfolio_stats._portfolio_stats(name=self.name)
            
            logger.info(f"Backtest completed for {self.name}")
            return results['portfolio'], self._performance_stats

        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            raise

    def get_performance_stats(self) -> Dict:
        """Get strategy performance statistics.

        Returns:
            Dictionary of performance metrics

        Raises:
            ValueError: If backtest hasn't been run
        """
        if self._performance_stats is None:
            raise ValueError("Must run backtest before getting performance stats")
        return self._performance_stats

    def plot_results(self, results: pd.DataFrame) -> None:
        """Plot backtest results including portfolio value and trades.

        Args:
            results: Portfolio results DataFrame from backtest
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(results.index, results['port_val'], label='Portfolio Value')
            
            # Add trade markers
            buy_points = results[results['Order'] == 'BUY'].index
            sell_points = results[results['Order'] == 'SELL'].index
            
            plt.scatter(buy_points, results.loc[buy_points, 'port_val'], 
                       marker='^', color='g', label='Buy')
            plt.scatter(sell_points, results.loc[sell_points, 'port_val'],
                       marker='v', color='r', label='Sell')
            
            plt.title(f'Strategy Results: {self.name}')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed - skipping plot generation")
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise