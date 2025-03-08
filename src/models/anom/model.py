"""
Anomaly detection model using various algorithms
"""
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats 
from logging import getLogger
from typing import Union, Optional, Dict, List
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parents[1]))
from connect import setup

logger = getLogger(__name__)

class anomaly_model(setup):
    """Anomaly detection model using various algorithms"""
    
    def __init__(self, df, feature_names, target_names, stock):
        """
        Initialize the anomaly detection model
        Args:
            df: DataFrame containing the data
            feature_names: List of feature column names
            target_names: List of target column names
            stock: Stock symbol
        """
        super().__init__(df, feature_names, target_names, stock)
        self.verbose = False
        self.models = {}
        self.model_training = {}
        self.training_preds = {}
        self.test_preds = {}
        self.decomp = {}
        self.decomp_preds = {}
        self.centers = {}
        self.distances = {}
        # Initialize data preprocessing
        self.initialize()
        
    def _isolation_forest(self):
        """Unsupervised Isolation Forest for anomaly detection"""
        if self.verbose:
            print("Running Isolation Forest...")
            
        iso = IsolationForest(contamination=0.01, random_state=42)
        iso.fit(self.xtrain)
        
        # Predict (-1 for outliers, 1 for inliers)
        train_pred = iso.predict(self.xtrain)
        test_pred = iso.predict(self.xtest)
        
        self.training_preds['IsolationForest'], self.test_preds['IsolationForest'] = \
            self.merge_preds(train_pred, test_pred, model_name="IsolationForest")
        
        self.models['IsolationForest'] = iso
        self.model_training['IsolationForest'] = iso.score_samples(self.xtrain)
            
    def _svm(self):
        """One Class SVM for anomaly detection"""
        if self.verbose:
            print("Running One-Class SVM...")
            
        svm = OneClassSVM(kernel='rbf', nu=0.1)
        svm.fit(self.xtrain)
        
        # Predict (-1 for outliers, 1 for inliers)
        train_pred = svm.predict(self.xtrain)
        test_pred = svm.predict(self.xtest)
        
        self.training_preds['SVM'], self.test_preds['SVM'] = \
            self.merge_preds(train_pred, test_pred, model_name="SVM")
        
        self.models['SVM'] = svm
        self.model_training['SVM'] = svm.score_samples(self.xtrain)
    
    def _lof(self):
        """Local Outlier Factor for anomaly detection"""
        if self.verbose:
            print("Running Local Outlier Factor...")
            
        lof = LocalOutlierFactor(n_neighbors=5, novelty=True, contamination=0.1)
        lof.fit(self.xtrain)
        
        # Predict (-1 for outliers, 1 for inliers)
        train_pred = lof.predict(self.xtrain)
        test_pred = lof.predict(self.xtest)
        
        self.training_preds['LOF'], self.test_preds['LOF'] = \
            self.merge_preds(train_pred, test_pred, model_name="LOF")
        
        self.models['LOF'] = lof
        self.model_training['LOF'] = lof.negative_outlier_factor_
    
    def _kmeansAnomalyDetection(self, x, threshold=1.1618, name='Kmeans'):
        """
        K-Means based anomaly detection
        Args:
            x: Input features
            threshold: Distance threshold for anomaly detection
            name: Model name for storing results
        Returns:
            DataFrame with predictions
        """
        if self.verbose:
            print(f"Running K-means anomaly detection ({name})...")
            
        # Find optimal number of clusters
        param_grid = {
            'n_clusters': np.arange(2, 6),
            'n_init': [10]  # Explicitly set n_init
        }
        kmeans = KMeans(random_state=42, n_init=10)  # Set n_init in base model
        grid_search = GridSearchCV(kmeans, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(x)
        
        # Use best parameters including n_init
        best_params = grid_search.best_params_
        if self.verbose:
            print(f"Best parameters: {best_params}")
            
        kmeans_model = KMeans(
            n_clusters=best_params['n_clusters'],
            n_init=best_params['n_init'],
            random_state=42
        )
        kmeans_model.fit(x)
        
        # Store model and get cluster assignments
        self.models[name] = kmeans_model
        centers = kmeans_model.cluster_centers_
        labels = kmeans_model.labels_
        
        # Calculate distances to cluster centers
        distances = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            point = x[i]
            center = centers[labels[i]]
            distances[i] = np.linalg.norm(point - center)
            
        self.distances[name] = distances
        
        # Convert distances to anomaly predictions (-1 for anomaly, 1 for normal)
        predictions = np.where(distances >= threshold, -1, 1)
        self.decomp_preds[name] = predictions
        self.centers[name] = labels
        
        return pd.DataFrame(predictions, index=self.features_scaled.index, columns=[name])
    
    def _find_optimal_threshold(self, distances, name):
        """
        Find optimal threshold for K-means anomaly detection
        Args:
            distances: Array of distances from points to cluster centers
            name: Model name for logging
        Returns:
            float: Optimal threshold value
        """
        thresholds = np.linspace(0, 2, 100)
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = (distances >= threshold).astype(int)
            score = f1_score(self.ytrain, predictions)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
                
        if self.verbose:
            print(f"{name} - Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
            
        return best_threshold
    
    def _pca(self):
        """PCA dimensionality reduction"""
        if self.verbose:
            print("Running PCA...")
            
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        transformed = pca.fit_transform(self.features_scaled)
        self.decomp['PCA'] = transformed
        return pca
    
    def _kernel_pca(self):
        """Kernel PCA dimensionality reduction"""
        if self.verbose:
            print("Running Kernel PCA...")
            
        kpca = KernelPCA(n_components=2, kernel='rbf')
        transformed = kpca.fit_transform(self.features_scaled)
        self.decomp['KPCA'] = transformed
        return kpca

    def _kmeans_pca(self, threshold=1.1618):
        """K-means anomaly detection in PCA space"""
        pca = self._pca()
        if self.verbose:
            print(f"Explained variance ratio: {pca.explained_variance_ratio_.cumsum()}")
            
        transformed = self.decomp['PCA']
        predictions = self._kmeansAnomalyDetection(transformed, threshold=threshold, name='PCA')
        
        # Store first two components with predictions
        components = pd.DataFrame(
            transformed[:, :2], 
            columns=['PC1', 'PC2'], 
            index=predictions.index
        )
        results = components.join(predictions)
        
        self.training_preds['PCA'], self.test_preds['PCA'] = self.pca_pred(predictions)
    
    def _kmeans_kernel_pca(self, threshold=1.1618):
        """K-means anomaly detection in Kernel PCA space"""
        self._kernel_pca()
        transformed = self.decomp['KPCA']
        predictions = self._kmeansAnomalyDetection(transformed, threshold=threshold, name='KPCA')
        
        components = pd.DataFrame(
            transformed[:, :2],
            columns=['PC1', 'PC2'],
            index=predictions.index
        )
        results = components.join(predictions)
        
        self.training_preds['KPCA'], self.test_preds['KPCA'] = self.pca_pred(predictions)
    
    def _ica_detection(self, threshold=1.1618):
        """Independent Component Analysis with K-means anomaly detection"""
        if self.verbose:
            print("Running ICA detection...")
            
        ica = FastICA(n_components=2, random_state=42)
        transformed = ica.fit_transform(self.features_scaled)
        self.decomp['ICA'] = transformed
        
        predictions = self._kmeansAnomalyDetection(transformed, threshold=threshold, name='ICA')
        
        components = pd.DataFrame(
            transformed,
            columns=['IC1', 'IC2'],
            index=predictions.index
        )
        results = components.join(predictions)
        
        self.training_preds['ICA'], self.test_preds['ICA'] = self.pca_pred(predictions)
    
    def fit(self, threshold=1.1618):
        """
        Fit all anomaly detection models
        Args:
            threshold: Distance threshold for K-means based methods
        """
        if not hasattr(self, 'features_scaled'):
            raise ValueError("Data not initialized. Call initialize() first.")
            
        # Traditional anomaly detection methods
        self._isolation_forest()
        self._svm()
        self._lof()
        
        # Dimensionality reduction + K-means methods
        self._kmeans_pca(threshold)
        self._kmeans_kernel_pca(threshold)
        self._ica_detection(threshold)
        
        self.decomp_models = ['PCA', 'KPCA', 'ICA']
        
        if self.verbose:
            print("\nModel fitting complete.")
            for name in self.models:
                print(f"{name} predictions shape - Train: {self.training_preds[name].shape}, Test: {self.test_preds[name].shape}")
