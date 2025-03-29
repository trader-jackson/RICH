import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import os

class RegimeDetector:
    """A class for detecting market regimes using HMM, GMM or clustering methods.
    
    This class implements methods to identify different market regimes (bull, bear, stable, etc.)
    using Hidden Markov Models, Gaussian Mixture Models, or clustering techniques on historical market data.
    """
    
    def __init__(self, n_regimes=4, method='hmm', window_size=252, step_size=21):
        """
        Initialize the RegimeDetector.
        
        Parameters:
        -----------
        n_regimes : int, default=4
            Number of regimes to detect.
        method : str, default='hmm'
            Method to use for regime detection ('hmm', 'gmm', or 'kmeans').
        window_size : int, default=252
            Size of the rolling window for feature calculation (252 trading days ≈ 1 year).
        step_size : int, default=21
            Step size for sliding window (21 trading days ≈ 1 month).
        """
        self.n_regimes = n_regimes
        self.method = method.lower()
        self.window_size = window_size
        self.step_size = step_size
        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = None
        self.regime_mapping = None  # For mapping numerical labels to meaningful names
        self.pca = None  # For PCA on the features if needed
        
    def _prepare_features(self, df):
        """
        Prepare features for regime detection.
        
        Here we compute rolling mean and standard deviation of log returns.
        
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
        
        # Apply PCA to reduce noise; here reduce to min(n_features, n_regimes) components.
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
            DataFrame containing price data with columns like ['date', 'open', 'high', 'low', 'close', 'volume'].
            
        Returns:
        --------
        self : RegimeDetector
            The fitted detector.
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
        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_regimes, 
                random_state=42,
                n_init=20
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
            raise ValueError(f"Unknown method: {self.method}. Use 'hmm', 'gmm' or 'kmeans'.")
        
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
        if self.method in ['hmm', 'gmm', 'kmeans']:
            regimes = self.model.predict(X)
        else:
            raise ValueError("Unknown method.")
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
        if self.method in ['hmm', 'gmm', 'kmeans']:
            regime = self.model.predict(features_pca)[0]
        else:
            raise ValueError("Unknown method.")
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
        Use a rolling window approach to compute the trend feature, and then cluster the windows based on this feature.
        
        For each window:
          - Trend = (end_price - start_price) / start_price
        
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
            DataFrame with computed trend and assigned cluster for each window.
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
            
            features.append([trend])  # Only using trend
            indices.append((start, end))
        
        features = np.array(features)
        
        # Cluster the windows based on trend only
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(features)
        
        # Store results in a DataFrame
        window_df = pd.DataFrame(features, columns=['trend'])
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

def save_regime_segments(df, col, method_name, output_dir="regime_label"):
    """
    Group the DataFrame by contiguous segments based on the column (e.g., 'regime')
    and save the start and end dates for each segment into CSV files.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame that contains at least 'date' and the specified column.
    col : str
        The column name to group by (e.g., 'regime').
    method_name : str
        A name to include in the filename (e.g., "gmm").
    output_dir : str, default="regime_label"
        The directory where files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Ensure data is sorted by date
    df = df.sort_values('date').reset_index(drop=True)
    # Create a grouping variable that increments when the regime changes
    df['segment_group'] = (df[col].shift() != df[col]).cumsum()
    segments = df.groupby(['segment_group', col]).agg(
        start_date=('date', 'first'),
        end_date=('date', 'last'),
        count=('date', 'count')
    ).reset_index()
    
    # Save one CSV file per regime label containing its segments
    for regime in segments[col].unique():
        regime_segments = segments[segments[col] == regime]
        out_file = os.path.join(output_dir, f"{method_name}_regime_{regime}_segments.csv")
        regime_segments[['start_date', 'end_date', 'count']].to_csv(out_file, index=False)
        print(f"Saved {method_name} regime {regime} segments to {out_file}")

if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.dates as mdates
    from statistics import mode, StatisticsError
    
    # Load data (ensure your CSV has 'date' and 'close' columns)
    df = pd.read_csv('market_index/raw/NASDAQ.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # ---------------------------
    # GMM method regime detection
    # ---------------------------
    gmm_detector = RegimeDetector(n_regimes=4, method='gmm')
    gmm_detector, gmm_labeled_df = gmm_detector.fit(df)
    
    # Instead of saving every date, group contiguous regimes into segments and record start/end dates
    save_regime_segments(gmm_labeled_df, col='regime', method_name="gmm")
    
    # (Optional) Plot the time series colored by GMM regime assignment
    plt.figure(figsize=(16, 8))
    for regime in range(gmm_detector.n_regimes):
        mask = gmm_labeled_df['regime'] == regime
        plt.plot(gmm_labeled_df['date'][mask], gmm_labeled_df['close'][mask], '.', 
                 label=f"{gmm_detector.get_regime_name(regime)} (Regime {regime})")
    plt.title('NASDAQ Market Regimes by GMM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # ---------------------------
    # Sliding Window Clustering using trend only
    # ---------------------------
    window_size = 100
    step = 20
    n_clusters = 4
    window_df, kmeans_model = gmm_detector.rolling_window_clustering(df, window_size=window_size, step=step, n_clusters=n_clusters)
    
    # For each time step in df, assign a cluster label based on overlapping windows.
    df_sorted = df.sort_values('date').reset_index(drop=True)
    n = len(df_sorted)
    cluster_assignments = []
    for i in range(n):
        labels_for_i = []
        for _, row in window_df.iterrows():
            if row['start_idx'] <= i < row['end_idx']:
                labels_for_i.append(row['cluster'])
        if labels_for_i:
            try:
                common_label = mode(labels_for_i)
            except StatisticsError:
                common_label = labels_for_i[0]
            cluster_assignments.append(common_label)
        else:
            cluster_assignments.append(np.nan)
    df_sorted['cluster'] = cluster_assignments
    
    # For the sliding window method, we can similarly group contiguous segments and save them.
    save_regime_segments(df_sorted, col='cluster', method_name="sliding")
    
    # Plot the entire time series with segments colored by sliding window cluster assignment
    df_sorted['cluster_group'] = (df_sorted['cluster'].shift() != df_sorted['cluster']).cumsum()
    cmap = plt.get_cmap('viridis', n_clusters)
    fig, ax = plt.subplots(figsize=(16, 8))
    for _, group in df_sorted.groupby('cluster_group'):
        clabel = group['cluster'].iloc[0]
        if pd.isna(clabel):
            continue
        ax.plot(group['date'], group['close'], color=cmap(int(clabel)),
                label=f"Cluster {int(clabel)}" if f"Cluster {int(clabel)}" not in ax.get_legend_handles_labels()[1] else "")
    ax.set_title("NASDAQ Time Series Colored by Sliding Window (Trend) Cluster Assignment")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()
