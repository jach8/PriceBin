import pickle
import pandas as pd 
import sqlite3 
import os 
import json 
import datetime as dt 
import numpy as np 

from typing import List, Dict, Union, Optional
from src.db_connect import Prices
from src.report import perf as performance
from src.utils import add_stock, delete_stock
 

def init():
    """ Initialize the program. """
    Initialize()
    
    
def get_path(pre=''):
    with open(pre + 'connections.json', 'r') as f:
        connections = json.load(f)
    return connections

def check_path(connections):
	# Check if the files exist
	checks = []
	for key, value in connections.items():
		if not os.path.exists(value):
			raise ValueError(f'{value} does not exist')
		checks.append(True)
	return all(checks)
           	 
class Manager:
	def __init__(self, connections=None):
		#  If type is string, or is None
		if type(connections) == str:
			connections = get_path(connections)
		if connections == None:
			connections = get_path()
		if type(connections) != dict:
			raise ValueError('Connections must be a dictionary')
		# Check if the files exist
		if check_path(connections) == False:
			raise ValueError('Files do not exist')

		# Initialize the Connections: 
		self.Pricedb = Prices(connections)
		# self.Earningsdb = Earnings(connections)
		self.performance = performance(connections) 
		# Save the Connection Dict.
		self.connection_paths = connections
		  
	def close_connection(self):
		self.Pricedb.close_connections()
  
	def addStock(self, stock):
		with sqlite3.connect(self.connection_paths['stock_names']) as conn:
		    add_stock(conn, self.connection_paths['ticker_path'], stock)
  
	def removeStock(self, stock):  
		with sqlite3.connect(self.connection_paths['stock_names']) as conn: 
			delete_stock(conn, self.connection_paths['ticker_path'], stock)
   
	def update_stock_prices(self):
		self.Pricedb.update_stocks()  


	
if __name__ == "__main__":
    m = Manager()
    m.update_stock_prices()
    m.close_connection()
 
 