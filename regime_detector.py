import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import os

class RegimeDetector:
    """A class for detecting market regimes using either HMM or clustering methods.
    
    This class implements methods to identify different market regimes (bull, bear, stable, etc.)
    using either Hidden Markov Models or clustering techniques on historical market data.
    """
    
    def __init__(self, n_regimes=4, method='hmm', window_size=252, step_size=21):
        """
        Initialize the RegimeDetector.
        
        Parameters:
        -----------
        n_regimes : int, default=4
            Number of regimes to detect (e.g., bull, bear, stable, volatile)
        method : str, default='hmm'
            Method to use for regime detection ('hmm' or 'kmeans')
        window_size : int, default=252
            Size of the rolling window for feature calculation (252 trading days ≈ 1 year)
        step_size : int, default=21
            Step size for sliding window (21 trading days ≈ 1 month)
        """
        self.n_regimes = n_regimes
        self.method = method.lower()
        self.window_size = window_size
        self.step_size = step_size
        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = None
        self.regime_mapping = None  # For mapping numerical labels to meaningful names
        self.pca = None  # For PCA on the features
        
    def _prepare_features(self, df):
        """
        Prepare features for regime detection.
        
        In this simplified version we only compute rolling mean and std on log returns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data with at least a 'close' column.
            
        Returns:
        --------
        X_pca : numpy.ndarray
            Array of features after PCA for regime detection.
        df : pandas.DataFrame
            DataFrame with newly added features.
        """
        # Calculate log returns if not already present
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close']).diff()
            
        # Calculate rolling statistics
        df['roll_mean'] = df['log_return'].rolling(window=self.window_size).mean()
        df['roll_std'] = df['log_return'].rolling(window=self.window_size).std()
        
        # Drop NaN values due to rolling calculations
        df = df.dropna()
        
        # Select features for regime detection
        features = ['roll_mean', 'roll_std']
        X = df[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA to reduce noise; reduce to min(n_features, n_regimes) components.
        n_components = min(len(features), self.n_regimes)
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        return X_pca, df
    
    def fit(self, df):
        """
        Fit the regime detection model to the data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data with columns ['date', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
        --------
        self : RegimeDetector
            The fitted detector
        df : pandas.DataFrame
            DataFrame with regime labels added.
        """
        X, df_clean = self._prepare_features(df)
        
        if self.method == 'hmm':
            self.model = GaussianHMM(
                n_components=self.n_regimes, 
                covariance_type='full', 
                n_iter=1000, 
                random_state=42
            )
            self.model.fit(X)
            regimes = self.model.predict(X)
        elif self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_regimes, 
                random_state=42, 
                n_init=20
            )
            regimes = self.model.fit_predict(X)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'hmm' or 'kmeans'.")
        
        df_clean['regime'] = regimes
        self.regime_labels = df_clean[['regime']]
       
        self._map_regimes_to_labels(df_clean)
        
        return self, df_clean
    
    def _map_regimes_to_labels(self, df):
        """
        Map numerical regime labels to meaningful names based on market characteristics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with regime labels and market data.
        """
        regime_stats = {}
        for regime in range(self.n_regimes):
            mask = df['regime'] == regime
            if mask.sum() > 0:
                avg_return = df.loc[mask, 'log_return'].mean() * 252  # Annualized
                volatility = df.loc[mask, 'log_return'].std() * np.sqrt(252)  # Annualized
                regime_stats[regime] = {'return': avg_return, 'volatility': volatility}
        
        self.regime_mapping = {}
        sorted_by_return = sorted(regime_stats.items(), key=lambda x: x[1]['return'], reverse=True)
        
        if len(sorted_by_return) >= 4:
            high_vol_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['volatility'], reverse=True)[:2]
            high_vol_ids = [r[0] for r in high_vol_regimes]
            
            for i, (regime, stats) in enumerate(sorted_by_return):
                if i == 0 and regime in high_vol_ids:
                    self.regime_mapping[regime] = 'bull_volatile'
                elif i == 0:
                    self.regime_mapping[regime] = 'bull_stable'
                elif i == len(sorted_by_return) - 1 and regime in high_vol_ids:
                    self.regime_mapping[regime] = 'bear_volatile'
                elif i == len(sorted_by_return) - 1:
                    self.regime_mapping[regime] = 'bear_stable'
                elif regime in high_vol_ids:
                    self.regime_mapping[regime] = 'volatile'
                else:
                    self.regime_mapping[regime] = 'stable'
        else:
            for i, (regime, stats) in enumerate(sorted_by_return):
                if i == 0:
                    self.regime_mapping[regime] = 'bull'
                elif i == len(sorted_by_return) - 1:
                    self.regime_mapping[regime] = 'bear'
                else:
                    self.regime_mapping[regime] = 'stable'
    
    def predict(self, df):
        """
        Predict regimes for new data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with regime labels.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X, df_clean = self._prepare_features(df)
        if self.method == 'hmm':
            regimes = self.model.predict(X)
        else:  # kmeans
            regimes = self.model.predict(X)
        df_clean['regime'] = regimes
        
        return df_clean
    
    def get_regime_label(self, state):
        """
        Get the regime label for a given state.
        
        Parameters:
        -----------
        state : dict or array-like
            The state observation containing keys/columns: 'roll_mean', 'roll_std'
            
        Returns:
        --------
        int
            The regime label.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features = np.array([state['roll_mean'], state['roll_std']]).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        if self.method == 'hmm':
            regime = self.model.predict(features_pca)[0]
        else:
            regime = self.model.predict(features_pca)[0]
        return regime
    
    def get_regime_name(self, regime_id):
        """
        Get the descriptive name for a regime ID.
        
        Parameters:
        -----------
        regime_id : int
            The numerical regime ID.
            
        Returns:
        --------
        str
            The descriptive name of the regime.
        """
        if self.regime_mapping is None:
            return f"Regime {regime_id}"
        return self.regime_mapping.get(regime_id, f"Regime {regime_id}")
    
    def rolling_window_clustering(self, df, window_size=100, step=20, n_clusters=4):
        """
        Use a rolling window approach to compute trend and volatility features,
        and then cluster the windows based on these features.
        
        For each window:
          - Trend = (end_price - start_price) / start_price
          - Volatility = standard deviation of daily returns within the window.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing at least 'date' and 'close' columns.
        window_size : int, default=100
            Number of data points in each window.
        step : int, default=20
            Number of data points to slide the window.
        n_clusters : int, default=4
            Number of clusters (market regimes) to form.
        
        Returns:
        --------
        window_df : pandas.DataFrame
            DataFrame with computed trend, volatility, and assigned cluster for each window.
        kmeans : KMeans
            The fitted KMeans clustering model.
        """
        # Ensure the DataFrame is sorted by date
        df_sorted = df.sort_values('date').reset_index(drop=True)
        features = []
        indices = []
        
        for start in range(0, len(df_sorted) - window_size + 1, step):
            end = start + window_size
            window = df_sorted.iloc[start:end]
            
            # Calculate trend: (end_price - start_price) / start_price
            start_price = window['close'].iloc[0]
            end_price = window['close'].iloc[-1]
            trend = (end_price - start_price) / start_price
            
            # Calculate volatility: standard deviation of daily returns in the window
            returns = window['close'].pct_change().dropna()
            volatility = returns.std()
            
            features.append([trend, volatility])
            indices.append((start, end))
        
        features = np.array(features)
        
        # Cluster the windows based on trend and volatility
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(features)
        
        # Store results in a DataFrame
        window_df = pd.DataFrame(features, columns=['trend', 'volatility'])
        window_df['cluster'] = labels
        window_df['start_idx'] = [idx[0] for idx in indices]
        window_df['end_idx'] = [idx[1] for idx in indices]
        return window_df, kmeans
    
    def save(self, filepath):
        """
        Save the fitted detector to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the detector.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'n_regimes': self.n_regimes,
                'method': self.method,
                'window_size': self.window_size,
                'step_size': self.step_size,
                'regime_mapping': self.regime_mapping,
                'pca': self.pca
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a fitted detector from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved detector.
            
        Returns:
        --------
        RegimeDetector
            The loaded detector.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        detector = cls(
            n_regimes=data['n_regimes'],
            method=data['method'],
            window_size=data['window_size'],
            step_size=data['step_size']
        )
        detector.model = data['model']
        detector.scaler = data['scaler']
        detector.regime_mapping = data['regime_mapping']
        detector.pca = data.get('pca', None)
        return detector

def label_data_with_regimes(df, save_path=None, method='hmm', n_regimes=4):
    """
    Label data with market regimes and optionally save the labeled data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data.
    save_path : str, optional
        Path to save the labeled data.
    method : str, default='hmm'
        Method to use for regime detection ('hmm' or 'kmeans').
    n_regimes : int, default=4
        Number of regimes to detect.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with regime labels.
    RegimeDetector
        The fitted detector.
    """
    detector = RegimeDetector(n_regimes=n_regimes, method=method)
    detector, labeled_df = detector.fit(df)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        labeled_df.to_csv(save_path, index=False)
        detector_path = os.path.join(os.path.dirname(save_path), 'regime_detector.pkl')
        detector.save(detector_path)
    
    return labeled_df, detector

if __name__ == "__main__":
    import yfinance as yf
    
    # Example usage: load data (ensure your CSV has 'date' and 'close' columns)
    df = pd.read_csv('market_index/raw/NASDAQ.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Use the original regime detection for demonstration
    detector = RegimeDetector(n_regimes=4, method='hmm')
    detector, labeled_df = detector.fit(df)
    
    plt.figure(figsize=(16, 8))
    for regime in range(detector.n_regimes):
        mask = labeled_df['regime'] == regime
        plt.plot(labeled_df.index[mask], labeled_df['close'][mask], '.', 
                 label=f"{detector.get_regime_name(regime)} (Regime {regime})")
    plt.title('NASDAQ Market Regimes')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Perform rolling window clustering based on trend and volatility
    window_df, kmeans_model = detector.rolling_window_clustering(df, window_size=100, step=20, n_clusters=4)
    print(window_df.head())
    
    # Plot the clustering results in feature space
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(window_df['trend'], window_df['volatility'], 
                          c=window_df['cluster'], cmap='viridis', s=100, edgecolors='k')
    plt.xlabel('Trend (End Price - Start Price) / Start Price')
    plt.ylabel('Volatility (Std of Daily Returns)')
    plt.title('Rolling Window Clustering Based on Trend and Volatility')
    plt.colorbar(scatter, label='Cluster')
    plt.show()
