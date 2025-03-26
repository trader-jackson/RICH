import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
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
        
    def _prepare_features(self, df):
        """
        Prepare features for regime detection.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data with at least 'close' column
            
        Returns:
        --------
        X : numpy.ndarray
            Array of features for regime detection
        """
        # Calculate log returns
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close']).diff()
            
        # Calculate rolling statistics
        df['roll_mean'] = df['log_return'].rolling(window=self.window_size).mean()
        df['roll_std'] = df['log_return'].rolling(window=self.window_size).std()
        df['momentum'] = df['close'].pct_change(periods=self.window_size)
        
        # Calculate True Range and ATR
        df['TR'] = np.maximum.reduce([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ])
        df['ATR'] = df['TR'].rolling(window=self.window_size).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        # Select features for regime detection
        features = ['roll_mean', 'roll_std', 'momentum', 'ATR']
        X = df[features].values
        
        return X, df
    
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
        """
        # Prepare features
        X, df_clean = self._prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'hmm':
            # Fit HMM model
            self.model = GaussianHMM(
                n_components=self.n_regimes, 
                covariance_type='full', 
                n_iter=1000, 
                random_state=42
            )
            self.model.fit(X_scaled)
            
            # Predict regimes
            regimes = self.model.predict(X_scaled)
            
        elif self.method == 'kmeans':
            # Fit KMeans model
            self.model = KMeans(
                n_clusters=self.n_regimes, 
                random_state=42, 
                n_init=20
            )
            regimes = self.model.fit_predict(X_scaled)
            
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'hmm' or 'kmeans'.")
        
        # Store regime labels
        df_clean['regime'] = regimes
        self.regime_labels = df_clean[['regime']]
        
        # Automatically map numerical regimes to meaningful labels based on characteristics
        self._map_regimes_to_labels(df_clean)
        
        return self
    
    def _map_regimes_to_labels(self, df):
        """
        Map numerical regime labels to meaningful names based on market characteristics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with regime labels and market data
        """
        # Calculate average return and volatility for each regime
        regime_stats = {}
        for regime in range(self.n_regimes):
            mask = df['regime'] == regime
            if mask.sum() > 0:
                avg_return = df.loc[mask, 'log_return'].mean() * 252  # Annualized
                volatility = df.loc[mask, 'log_return'].std() * np.sqrt(252)  # Annualized
                regime_stats[regime] = {
                    'return': avg_return,
                    'volatility': volatility
                }
        
        # Define regime mapping based on return and volatility characteristics
        self.regime_mapping = {}
        
        # Sort regimes by return (descending)
        sorted_by_return = sorted(regime_stats.items(), key=lambda x: x[1]['return'], reverse=True)
        
        # Assign labels based on return and volatility
        if len(sorted_by_return) >= 4:
            # If we have 4 or more regimes, we can be more specific
            high_vol_regimes = sorted([r for r in regime_stats.items()], key=lambda x: x[1]['volatility'], reverse=True)[:2]
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
            # Simpler mapping for fewer regimes
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
            DataFrame containing price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with regime labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        X, df_clean = self._prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict regimes
        if self.method == 'hmm':
            regimes = self.model.predict(X_scaled)
        else:  # kmeans
            regimes = self.model.predict(X_scaled)
        
        # Add regime labels to DataFrame
        df_clean['regime'] = regimes
        
        return df_clean
    
    def get_regime_label(self, state):
        """
        Get the regime label for a given state.
        
        Parameters:
        -----------
        state : array-like
            The state observation
            
        Returns:
        --------
        int
            The regime label
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract relevant features from state and scale them
        features = np.array([state['roll_mean'], state['roll_std'], 
                            state['momentum'], state['ATR']]).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Predict regime
        if self.method == 'hmm':
            regime = self.model.predict(features_scaled)[0]
        else:  # kmeans
            regime = self.model.predict(features_scaled)[0]
        
        return regime
    
    def get_regime_name(self, regime_id):
        """
        Get the descriptive name for a regime ID.
        
        Parameters:
        -----------
        regime_id : int
            The numerical regime ID
            
        Returns:
        --------
        str
            The descriptive name of the regime
        """
        if self.regime_mapping is None:
            return f"Regime {regime_id}"
        return self.regime_mapping.get(regime_id, f"Regime {regime_id}")
    
    def sliding_window_detection(self, df):
        """
        Detect regimes using a sliding window approach.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with regime labels for each window
        """
        # Prepare features
        X, df_clean = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize array for storing regime labels
        all_regimes = np.full(len(X_scaled), np.nan)
        
        # Sliding window regime detection
        n_windows = (len(X_scaled) - self.window_size) // self.step_size + 1
        
        for i in range(n_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            window_X = X_scaled[start_idx:end_idx]
            
            if self.method == 'hmm':
                model = GaussianHMM(
                    n_components=self.n_regimes, 
                    covariance_type='full', 
                    n_iter=300, 
                    random_state=42
                )
                model.fit(window_X)
                window_regimes = model.predict(window_X)
            else:  # kmeans
                model = KMeans(
                    n_clusters=self.n_regimes, 
                    random_state=42, 
                    n_init=10
                )
                window_regimes = model.fit_predict(window_X)
            
            # Store regime labels
            if i == 0:
                all_regimes[start_idx:end_idx] = window_regimes
            else:
                all_regimes[end_idx-self.step_size:end_idx] = window_regimes[-self.step_size:]
        
        # Add regime labels to DataFrame
        df_clean['regime'] = all_regimes
        
        return df_clean
    
    def save(self, filepath):
        """
        Save the fitted detector to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the detector
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
                'regime_mapping': self.regime_mapping
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a fitted detector from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved detector
            
        Returns:
        --------
        RegimeDetector
            The loaded detector
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
        
        return detector


def label_data_with_regimes(df, save_path=None, method='hmm', n_regimes=4):
    """
    Label data with market regimes and optionally save the labeled data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data
    save_path : str, optional
        Path to save the labeled data
    method : str, default='hmm'
        Method to use for regime detection ('hmm' or 'kmeans')
    n_regimes : int, default=4
        Number of regimes to detect
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with regime labels
    RegimeDetector
        The fitted detector
    """
    # Create and fit regime detector
    detector = RegimeDetector(n_regimes=n_regimes, method=method)
    labeled_df = detector.fit(df)
    
    # Save labeled data if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        labeled_df.to_csv(save_path, index=False)
        
        # Save detector alongside data
        detector_path = os.path.join(os.path.dirname(save_path), 'regime_detector.pkl')
        detector.save(detector_path)
    
    return labeled_df, detector


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    sp500 = yf.download('^GSPC', start='2010-01-01', end='2020-12-31')
    
    # Detect regimes
    detector = RegimeDetector(n_regimes=4, method='hmm')
    labeled_df = detector.fit(sp500)
    
    # Plot regimes
    plt.figure(figsize=(16, 8))
    for regime in range(detector.n_regimes):
        mask = labeled_df['regime'] == regime
        plt.plot(labeled_df.index[mask], labeled_df['close'][mask], '.', 
                 label=f"{detector.get_regime_name(regime)} (Regime {regime})")
    
    plt.title('S&P 500 Market Regimes')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()