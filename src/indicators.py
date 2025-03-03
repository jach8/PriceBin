import pandas as pd 
import numpy as np 
import sqlite3 as sql
import logging
from typing import Union, Tuple, Dict, List, Optional
from pandas import Series, DataFrame
from technicals.vol import volatility 
from technicals.others import descriptive_indicators
from technicals.ma import moving_avg
# from .technicals.vol import volatility 
# from .technicals.others import descriptive_indicators
# from .technicals.ma import moving_avg

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
        self.moving_average.windows = np.array([10, 20, 50, 100, 200])
        self.volatility.windows = np.array([6, 10, 20, 28])
        self.descriptive.windows = np.array([10, 20])
    
    def moving_average_ribbon(self, df: pd.DataFrame, ma:str='sma') -> pd.DataFrame:
        """ Generate a moving average ribbon """
        assert ma in ['sma', 'ema', 'wma', 'kama'], 'Invalid moving average type'
        print(type(df))
        df = self.moving_average._validate_dataframe(df)
        return self.moving_average.ribbon(df, ma=ma)
    
    def volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Generate volatility indicators """
        df = self.volatility._validate_dataframe(df)
        return self.volatility.vol_indicators(df)
    






if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from main import Manager, get_path 

    connections = get_path()
    m = Manager(connections)
    df = m.Pricedb.ohlc('spy', daily=False).resample('3T').last().drop(columns = ['Date'])
    df = m.Pricedb.ohlc('spy', daily=True).resample('3D').last().drop(columns = ['Date'])
    print(df)
    i = Indicators()
    print(i.moving_average_ribbon(df, ma='sma'))
