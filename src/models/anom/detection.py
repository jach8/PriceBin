import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import IsolationForest 
import scipy.spatial.distance as ssd 
import scipy.sparse.csgraph as cg
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split, KFold

class anomaly_detection:
    def __init__(self):
        self.verbose = False
        pass
    
    def split(X, y, testsize = 0.2):
        return train_test_split(X, y, test_size = testsize, shuffle = False, )
        
    def iso_model(self, xtrain, xtest, contamination = 'auto'):
        ''' 
        Fit Isolation Forest model to data to find anomalies within the features. 
        
        Parameters:
            x_train: training data
            x_test: testing data
        
        Returns:
            y_pred: predictions of anomalies (-1 for anomaly, 1 for normal)

        '''
        # param_grid = dict(contamination = np.linspace(0.01, 0.1, 10))
        # model = IsolationForest()
        # model_cv = GridSearchCV(model, param_grid, cv=5)
        # model_cv.fit(xtrain)
        # self.model_cv = model_cv
        # return model_cv.predict(xtest)
        model = IsolationForest(contamination = contamination, max_features =len(list(xtrain.columns)), bootstrap = True, n_estimators = 100, random_state = 1)
        model.fit(xtrain)
        self.model = model
        return model.predict(xtest)
    
    def PCA(self, X, k = 2):
        '''
        PCA: Principal Component Analysis
        Inputs:
            X: n x m matrix, n is the number of samples, m is the number of features
            k: the number of components to keep
        Algorithm:
            1. Center the data X - µ 
            2. Estimate the covariance matrix C = X^T X / n 
            3. Compute the eigenvectors W and eigenvalues S of C, W = C dot V, S = C dot U
            4. Sort eigenvectors by eigenvalues in descending order
        Returns: 
            Z: The Projection of X onto the eigenvectors
        '''
        mu = np.mean(X, axis=0)
        X_centered = X - mu
        C = np.dot(X_centered, X_centered.T) / X.shape[0] # covariance matrix
        S, W = np.linalg.eig(C) # S: eigenvalues, W: eigenvectors
        S = S.real; W = W.real 
        idx = S.argsort()[::-1]
        S = S[idx]; W = W[:, idx] # sort eigenvalues and eigenvectors
        W = W[:, :k] # select the first k eigenvectors
        Z = np.dot(W.T, X_centered) # project X onto the first k eigenvectors
        
        return Z.T
    
    def pca_anom(self, X, k = 2, threshold = 1.618):
        ''' 
        Fit PCA to the data and find anomalies using KMeans clustering.
        Parameters:
            X: data to fit
            k: number of components to keep
        Returns:
            y_pred: predictions of anomalies (-1 for anomaly, 1 for normal)
        '''
        model_pc = self.PCA(X, k)
        param_grid = {'n_clusters': np.arange(2, 10)}
        kmeans = KMeans()
        kmeans_cv = GridSearchCV(kmeans, param_grid, cv=5)
        kmeans_cv.fit(model_pc)
        if self.verbose:
            print("Best Score: ", kmeans_cv.best_score_)
        kmeans_model = KMeans(n_clusters=kmeans_cv.best_params_['n_clusters'], random_state=1).fit(model_pc)
        centers = kmeans_model.cluster_centers_
        labs = kmeans_model.labels_
        points = model_pc
        distances = np.zeros((len(points)))
        anoms = (distances >= threshold).astype(int)
        anoms = np.where(anoms == 1, -1, 1)
        a = pd.Series(anoms, index = X.index)
        return a