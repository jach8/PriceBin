import pandas as pd 
import numpy as np 
import sqlite3 as sql
import logging
from typing import Union, Tuple, Dict, List, Optional
from pandas import Series, DataFrame
from .technicals.vol import volatility 
from .technicals.others import descriptive_indicators
from .technicals.ma import moving_avg

# Configure logging
logging.basicConfig(
    # filename='logs/indicators.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Indicators:
    def __init__(self, *args, **kwargs) -> None:
        self.moving_average = moving_avg()
        self.volatility = volatility()
        self.descriptive = descriptive_indicators()
        self.ma_windows = np.array([10, 20, 50, 100, 200])
        self.vol_windows = np.array([6, 10, 20, 28])
        self.other_windows = np.array([10, 20])

    def _validate_dataframe(self, df:pd.DataFrame) -> pd.DataFrame:
        """ Validate the OHLCV DataFrame """
        return self.moving_average._validate_dataframe(df)
    
    






if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from main import Manager, get_path 

    connections = get_path()
    m = Manager(connections)
    df = m.Pricedb.ohlc('spy', daily=False).resample('3T').last()

    print(df)