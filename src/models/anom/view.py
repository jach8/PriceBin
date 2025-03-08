import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import sys
from .model import anomaly_model as models

class viewer(models):
    def __init__(self, df, feature_names, target_names, stock, verbose = False):
        super().__init__(df, feature_names, target_names, stock, verbose)
        
        
    def general_plot(self):
        all_preds = list(self.training_preds.keys())
        n = len(all_preds)
        fig, ax = plt.subplots(3, 3, figsize = (20, 6))
        ax = ax.flatten()
        ii = 0 
        for i, p in enumerate(list(self.decomp.keys())[:]):
            ax[ii].scatter(self.decomp[p][:, 0], self.decomp[p][:, 1], c = self.decomp_preds[p])
            ax[ii].set_title(f'{p} Anomalies')
            ii += 1
        print(ii == 3)

        for i, p in enumerate(all_preds[:]):
            cs = self.test_preds[p]['close']
            sc = self.test_preds[p]
            ax[ii].plot(self.test_preds[p]['close'], label = 'close')
            ax[ii].scatter(self.test_preds[p].index, self.test_preds[p]['close'], c = self.test_preds[p][p])
            ax[ii].set_title(f'{p} Anomalies')
            ax[ii].legend()
            ii += 1
                    
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        all_preds = list(self.training_preds.keys())
        lodf = []
        for i in all_preds:
            # Append the last test prediction
            lodf.append(self.test_preds[i][i])

        self.last_pred = pd.concat(lodf, axis = 1, keys = all_preds).tail(1)
        