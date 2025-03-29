import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import TRAIN_START_DATE, TRAIN_END_DATE

class MarketClusterer:
    """
    A simple class for clustering market regimes using a sliding window approach.
    
    For each window (of fixed size), this class computes:
      - Cumulative return: (end_price - start_price) / start_price
      - Volatility: Standard deviation of log returns within the window
      
    Then the windows are split into three groups by return (bottom 33%, middle 33%, top 33%).
    Within each return group, they are further split by volatility (bottom 33%, middle 33%, top 33%).
    The combined group index (0-8) is used as the cluster label.
    """
    
    def __init__(self, window_size=100, step=20):
        self.window_size = window_size
        self.step = step
    
    def rolling_window_clustering(self, df):
        """
        Compute the cumulative return and volatility for each sliding window and assign a cluster label.
        
        Parameters:
            df (pandas.DataFrame): DataFrame with at least 'date' and 'close' columns.
        
        Returns:
            window_df (pandas.DataFrame): DataFrame containing:
                'return', 'volatility', 'cluster',
                'start_idx', 'end_idx', 'start_date', and 'end_date'
                for each valid sliding window.
        """
        # Ensure the DataFrame is sorted by date
        df_sorted = df.sort_values('date').reset_index(drop=True)
        features = []
        indices = []
        date_ranges = []  # store actual start and end dates
        
        for start in range(0, len(df_sorted) - self.window_size + 1, self.step):
            end = start + self.window_size
            window = df_sorted.iloc[start:end]
            # Ensure the window is exactly the desired length
            if len(window) != self.window_size:
                continue
            
            # Record actual dates
            start_date = window['date'].iloc[0]
            end_date = window['date'].iloc[-1]
            date_ranges.append((start_date, end_date))
            
            # Compute cumulative return over the window
            start_price = window['close'].iloc[0]
            end_price = window['close'].iloc[-1]
            window_return = (end_price - start_price) / start_price
            
            # Compute volatility as the standard deviation of log returns
            log_returns = np.log(window['close']).diff().dropna()
            window_volatility = log_returns.std() if not log_returns.empty else 0.0
            
            features.append([window_return, window_volatility])
            indices.append((start, end))
        
        if not features:
            return pd.DataFrame()  # No valid windows found
        
        features = np.array(features)
        returns = features[:, 0]
        volatilities = features[:, 1]
        
        # Compute return percentiles (33rd and 66th)
        r33, r66 = np.percentile(returns, [33, 66])
        
        # For each return group, gather volatility values to compute volatility percentiles
        groups = {0: [], 1: [], 2: []}
        for i, ret in enumerate(returns):
            if ret < r33:
                groups[0].append(volatilities[i])
            elif ret < r66:
                groups[1].append(volatilities[i])
            else:
                groups[2].append(volatilities[i])
        
        # Compute volatility percentiles for each return group
        vol_percentiles = {}
        for grp in groups:
            vols = groups[grp]
            if len(vols) > 0:
                v33, v66 = np.percentile(vols, [33, 66])
                vol_percentiles[grp] = (v33, v66)
            else:
                vol_percentiles[grp] = (None, None)
                
        # Assign a combined cluster label for each window based on return and volatility groups
        cluster_assignments = []
        for i in range(len(features)):
            ret = returns[i]
            vol = volatilities[i]
            # Determine return group
            if ret < r33:
                r_group = 0
            elif ret < r66:
                r_group = 1
            else:
                r_group = 2
            v33, v66 = vol_percentiles[r_group]
            if v33 is None:
                v_group = 0
            else:
                if vol < v33:
                    v_group = 0
                elif vol < v66:
                    v_group = 1
                else:
                    v_group = 2
            # Combined label (0-8)
            label = r_group * 3 + v_group
            cluster_assignments.append(label)
        
        # Construct the output DataFrame
        window_df = pd.DataFrame(features, columns=['return', 'volatility'])
        window_df['cluster'] = cluster_assignments
        window_df['start_idx'] = [idx[0] for idx in indices]
        window_df['end_idx'] = [idx[1] for idx in indices]
        window_df['start_date'] = [d[0] for d in date_ranges]
        window_df['end_date'] = [d[1] for d in date_ranges]
        
        return window_df

def save_window_segments(window_df, output_dir):
    """
    Save each sliding window segment (each exactly the window size) as its own record.
    The saved CSV file will include a descriptive label for each window.
    
    Parameters:
        window_df (pandas.DataFrame): DataFrame with window information.
        output_dir (str): Directory where CSV file will be saved.
    """
    if window_df.empty:
        print("No valid windows found. Skipping save.")
        return
    
    # Define descriptive label mappings
    trend_mapping = {0: "Low Trend", 1: "Medium Trend", 2: "High Trend"}
    vol_mapping = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
    
    # Add a descriptive label column based on the numeric cluster
    window_df["desc_cluster"] = window_df["cluster"].apply(
        lambda x: f"{trend_mapping[x // 3]}, {vol_mapping[x % 3]}"
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sliding_windows_segments.csv")
    window_df.to_csv(output_path, index=False)
    print(f"Saved sliding window segments to {output_path}")

def process_dataset(csv_file, window_size=100, step=20, show_plot=True):
    """
    Process a single dataset (CSV file), performing:
      1. Loading and filtering by TRAIN_START_DATE/END_DATE
      2. Clustering using a sliding window approach
      3. Saving cluster segments
      4. Plotting one representative window per cluster (if show_plot=True)
    """
    # 1. Load & Filter
    df = pd.read_csv(os.path.join("market_index", "raw", csv_file))
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] < TRAIN_END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)
    
    # 2. Apply sliding window clustering
    clusterer = MarketClusterer(window_size=window_size, step=step)
    window_df = clusterer.rolling_window_clustering(df)
    
    # Create output directory for this dataset, e.g. "regime_label/Nasdaq100"
    dataset_name = os.path.splitext(csv_file)[0]
    output_dir = os.path.join("regime_label", dataset_name)
    save_window_segments(window_df, output_dir)
    
    # Print cluster statistics
    if not window_df.empty:
        cluster_stats = window_df.groupby("cluster").agg(
            mean_return=("return", "mean"), 
            mean_volatility=("volatility", "mean")
        ).reset_index()
        
        trend_mapping = {0: "Low Trend", 1: "Medium Trend", 2: "High Trend"}
        vol_mapping = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
        
        print(f"\nCluster Statistics for {csv_file}:")
        for _, row in cluster_stats.iterrows():
            cluster_label = int(row["cluster"])
            r_group = cluster_label // 3
            v_group = cluster_label % 3
            descriptive_label = f"{trend_mapping[r_group]}, {vol_mapping[v_group]}"
            print(f"  Cluster {cluster_label}: {descriptive_label} "
                  f"- Mean Return: {row['mean_return']:.4f}, "
                  f"Mean Volatility: {row['mean_volatility']:.4f}")
    
    # 4. (Optional) Plot a representative window for each cluster
    if show_plot and not window_df.empty:
        df_sorted = df.reset_index(drop=True)
        representative_windows = {}
        for cluster in range(9):  # clusters 0 to 8
            cluster_windows = window_df[window_df['cluster'] == cluster]
            if not cluster_windows.empty:
                # Choose the first window as a representative example
                representative_windows[cluster] = cluster_windows.iloc[0]
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=False, sharey=False)
        axes = axes.flatten()
        
        for cluster, rep_row in representative_windows.items():
            ax = axes[cluster]
            start_idx = int(rep_row['start_idx'])
            end_idx = int(rep_row['end_idx'])
            segment = df_sorted.iloc[start_idx:end_idx]
            # Decode descriptive label
            r_group = cluster // 3
            v_group = cluster % 3
            descriptive_label = f"{trend_mapping[r_group]}, {vol_mapping[v_group]}"
            start_date = segment['date'].iloc[0].strftime('%Y-%m-%d')
            end_date = segment['date'].iloc[-1].strftime('%Y-%m-%d')
            ax.plot(segment['date'], segment['close'])
            ax.set_title(f"Cluster {cluster}:\n{descriptive_label}\n{start_date} to {end_date}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)
        
        # Hide any unused subplots (if any clusters are missing)
        for i in range(len(representative_windows), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f"Representative Windows for Each Cluster ({dataset_name}, window={window_size})",
                     fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    # List of CSV files to process
    csv_files = ["Dow30.csv", "Nasdaq100.csv", "SSE50.csv"]
    
    for file in csv_files:
        print(f"\nProcessing {file} ...")
        process_dataset(csv_file=file, window_size=100, step=20, show_plot=True)
