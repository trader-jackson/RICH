import os
import numpy as np
import pandas as pd
import torch
import pickle
import argparse
from stable_baselines.common.vec_env import DummyVecEnv

from regime_detector import RegimeDetector, label_data_with_regimes
from hierarchical_agent import HierarchicalAgent, SubAgent, create_regime_specific_env
from env_stocktrading_forecasting import StockTradingEnv
from DRLAgent import DRLAgent
import config

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create directories
os.makedirs('trained_models/hierarchical', exist_ok=True)
os.makedirs('results/hierarchical', exist_ok=True)
os.makedirs('tensorboard_log/hierarchical', exist_ok=True)
os.makedirs('regime_boundaries', exist_ok=True)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train hierarchical agent with market data')
    parser.add_argument('--market', type=str, default='us_stocks', choices=['us_stocks', 'cn_stocks'],
                        help='Market to use (us_stocks, cn_stocks)')
    parser.add_argument('--index', type=str, default='dow30', choices=['dow30', 'nasdaq100', 'hs300'],
                        help='Index to use (dow30, nasdaq100, hs300)')
    parser.add_argument('--n_regimes', type=int, default=4,
                        help='Number of market regimes to detect')
    parser.add_argument('--method', type=str, default='hmm', choices=['hmm', 'kmeans'],
                        help='Method for regime detection (hmm, kmeans)')
    return parser.parse_args()

# Load stock data from directory
def load_stock_data(market, index):
    """Load all stock data from the specified market and index."""
    data_dir = f"data/{market}/{index}/processed"
    stock_list_path = f"data/{market}/{index}/stock_list.txt"
    
    # Read stock list
    with open(stock_list_path, 'r') as f:
        tickers = [line.strip() for line in f.readlines()]
    
    # Load data for each ticker
    stock_data = {}
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Ensure date is in datetime format
            if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Add ticker column if not present
            if 'tic' not in df.columns:
                df['tic'] = ticker
            
            stock_data[ticker] = df
    
    return stock_data, tickers

# Create market index data by averaging stock returns
def create_market_index(stock_data, tickers):
    """Create market index data by averaging stock returns."""
    # Get common date range
    all_dates = set()
    for ticker in tickers:
        if ticker in stock_data:
            all_dates.update(stock_data[ticker]['date'].tolist())
    
    common_dates = sorted(all_dates)
    
    # Create index DataFrame
    index_data = pd.DataFrame({'date': common_dates})
    index_data = index_data.set_index('date')
    
    # Calculate average OHLCV values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        values = {}
        for date in common_dates:
            date_values = []
            for ticker in tickers:
                if ticker in stock_data:
                    ticker_df = stock_data[ticker]
                    if date in ticker_df['date'].values:
                        row = ticker_df[ticker_df['date'] == date]
                        if not row.empty and col in row.columns:
                            date_values.append(row[col].values[0])
            
            if date_values:
                values[date] = np.mean(date_values)
        
        index_data[col] = pd.Series(values)
    
    # Reset index to get date as column
    index_data = index_data.reset_index()
    
    # Fill any missing values using forward fill
    index_data = index_data.fillna(method='ffill').fillna(method='bfill')
    
    return index_data

# Split data into train, validation, and test sets
def split_data(df):
    """Split data into train, validation, and test sets based on date."""
    # Ensure date is in datetime format
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Split data based on date ranges in config
    train_start_date = pd.to_datetime(config.TRAIN_START_DATE)
    train_end_date = pd.to_datetime(config.TRAIN_END_DATE)
    eval_start_date = pd.to_datetime(config.EVAL_START_DATE)
    eval_end_date = pd.to_datetime(config.EVAL_END_DATE)
    test_start_date = pd.to_datetime(config.TEST_START_DATE)
    test_end_date = pd.to_datetime(config.TEST_END_DATE)
    
    train_data = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)]
    eval_data = df[(df['date'] >= eval_start_date) & (df['date'] <= eval_end_date)]
    test_data = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]
    
    return train_data, eval_data, test_data

# Detect market regimes using index data
def detect_regimes_with_index(index_data, n_regimes=4, method='hmm'):
    """Detect market regimes using index data."""
    # Split index data
    train_index, eval_index, test_index = split_data(index_data)
    
    # Create and fit regime detector on training data
    detector = RegimeDetector(n_regimes=n_regimes, method=method)
    train_index = detector.fit(train_index)
    
    # Predict regimes for evaluation and test data
    eval_index = detector.predict(eval_index)
    test_index = detector.predict(test_index)
    
    # Print regime statistics
    print("\nRegime Statistics (Training Data):")
    for regime in range(n_regimes):
        regime_name = detector.get_regime_name(regime)
        count = (train_index['regime'] == regime).sum()
        percentage = count / len(train_index) * 100
        print(f"  {regime_name}: {count} samples ({percentage:.2f}%)")
    
    # Save regime boundaries
    regime_boundaries = {}
    for regime in range(n_regimes):
        regime_name = detector.get_regime_name(regime)
        regime_data = train_index[train_index['regime'] == regime]
        if not regime_data.empty:
            regime_boundaries[regime_name] = {
                'start_dates': regime_data['date'].tolist(),
                'end_dates': regime_data['date'].tolist(),
                'avg_return': regime_data['log_return'].mean() if 'log_return' in regime_data.columns else None,
                'volatility': regime_data['log_return'].std() if 'log_return' in regime_data.columns else None
            }
    
    # Save regime boundaries to file
    os.makedirs('regime_boundaries', exist_ok=True)
    with open(f'regime_boundaries/regimes_{method}_{n_regimes}.pkl', 'wb') as f:
        pickle.dump(regime_boundaries, f)
    
    return detector, train_index, eval_index, test_index, regime_boundaries

# Label stock data with regimes from index
def label_stocks_with_regimes(stock_data, index_data_with_regimes):
    """Label stock data with regimes from index data."""
    labeled_stock_data = {}
    
    # Create a mapping of date to regime
    date_to_regime = dict(zip(index_data_with_regimes['date'], index_data_with_regimes['regime']))
    
    # Label each stock's data with regimes
    for ticker, df in stock_data.items():
        df_copy = df.copy()
        df_copy['regime'] = df_copy['date'].map(date_to_regime)
        
        # Fill any missing regime values with forward fill
        df_copy['regime'] = df_copy['regime'].fillna(method='ffill').fillna(method='bfill')
        
        labeled_stock_data[ticker] = df_copy
    
    return labeled_stock_data

# Create multi-stock environment
def create_multi_stock_env(stock_data, tickers, mode='train'):
    """Create a multi-stock trading environment."""
    # Combine all stock data into a single DataFrame
    all_data = []
    for ticker in tickers:
        if ticker in stock_data:
            all_data.append(stock_data[ticker])
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Environment parameters
    env_params = {
        'stock_dim': len(tickers),  # Number of stocks
        'hmax': 100,
        'initial_amount': 1000000,
        'transaction_cost_pct': 0.001,
        'reward_scaling': 1e-4,
        'state_space': 1,
        'action_space': len(tickers),  # Action for each stock
        'tech_indicator_list': config.TECHICAL_INDICATORS,
        'temporal_feature_list': config.TEMPORAL_FEATURE,
        'additional_list': [],
        'time_window_start': [0],  # Starting index
        'episode_length': 252,  # One year of trading days
        'temporal_len': 60,
        'hidden_channel': 4,
        'short_prediction_model_path': None,
        'long_prediction_model_path': None,
        'make_plots': False,
        'print_verbosity': 0,
        'iteration': 0
    }
    
    # Create environment
    env = StockTradingEnv(combined_df, **env_params, mode=mode)
    
    return env

# Create regime-specific environments for each stock
def create_regime_specific_envs(labeled_stock_data, tickers, detector, regime_names):
    """Create trading environments for each regime and stock."""
    # Split data for each stock
    train_data = {}
    eval_data = {}
    test_data = {}
    
    for ticker in tickers:
        if ticker in labeled_stock_data:
            train_ticker, eval_ticker, test_ticker = split_data(labeled_stock_data[ticker])
            train_data[ticker] = train_ticker
            eval_data[ticker] = eval_ticker
            test_data[ticker] = test_ticker
    
    # Environment parameters
    env_params = {
        'stock_dim': len(tickers),
        'hmax': 100,
        'initial_amount': 1000000,
        'transaction_cost_pct': 0.001,
        'reward_scaling': 1e-4,
        'state_space': 1,
        'action_space': len(tickers),
        'tech_indicator_list': config.TECHICAL_INDICATORS,
        'temporal_feature_list': config.TEMPORAL_FEATURE,
        'additional_list': [],
        'time_window_start': [0],
        'episode_length': 252,
        'temporal_len': 60,
        'hidden_channel': 4,
        'short_prediction_model_path': None,
        'long_prediction_model_path': None,
        'make_plots': False,
        'print_verbosity': 0,
        'iteration': 0
    }
    
    # Create environments for each regime
    train_envs = {}
    eval_envs = {}
    test_envs = {}
    
    for regime_name in regime_names:
        # Filter data for this regime
        regime_train_data = {}
        regime_eval_data = {}
        regime_test_data = {}
        
        for ticker in tickers:
            if ticker in train_data:
                # Get regime ID for the given name
                regime_id = None
                for rid, rname in detector.regime_mapping.items():
                    if rname == regime_name:
                        regime_id = rid
                        break
                
                if regime_id is not None:
                    # Filter data for this regime
                    regime_train_data[ticker] = train_data[ticker][train_data[ticker]['regime'] == regime_id]
                    regime_eval_data[ticker] = eval_data[ticker][eval_data[ticker]['regime'] == regime_id]
                    regime_test_data[ticker] = test_data[ticker][test_data[ticker]['regime'] == regime_id]
        
        # Combine regime data for all stocks
        combined_train = pd.concat(list(regime_train_data.values()), ignore_index=True) if regime_train_data else pd.DataFrame()
        combined_eval = pd.concat(list(regime_eval_data.values()), ignore_index=True) if regime_eval_data else pd.DataFrame()
        combined_test = pd.concat(list(regime_test_data.values()), ignore_index=True) if regime_test_data else pd.DataFrame()
        
        if not combined_train.empty:
            # Create environments
            train_env = StockTradingEnv(combined_train, **env_params, mode='train')
            train_envs[regime_name] = train_env
            
            if not combined_eval.empty:
                eval_env = StockTradingEnv(combined_eval, **env_params, mode='eval')
                eval_envs[regime_name] = eval_env
            
            if not combined_test.empty:
                test_env = StockTradingEnv(combined_test, **env_params, mode='test')
                test_envs[regime_name] = test_env
    
    # Create full environments for meta-agent training and testing
    full_train_data = pd.concat(list(train_data.values()), ignore_index=True) if train_data else pd.DataFrame()
    full_eval_data = pd.concat(list(eval_data.values()), ignore_index=True) if eval_data else pd.DataFrame()
    full_test_data = pd.concat(list(test_data.values()), ignore_index=True) if test_data else pd.DataFrame()
    
    full_train_env = StockTradingEnv(full_train_data, **env_params, mode='train') if not full_train_data.empty else None
    full_eval_env = StockTradingEnv(full_eval_data, **env_params, mode='eval') if not full_eval_data.empty else None
    full_test_env = StockTradingEnv(full_test_data, **env_params, mode='test') if not full_test_data.empty else None
    
    return train_envs, eval_envs, test_envs, full_train_env, full_eval_env, full_test_env

# Train subagents for each regime
def train_subagents(train_envs, eval_envs, regime_names):
    """Train specialized subagents for each market regime."""
    subagents = {}
    
    for regime_name in regime_names:
        if regime_name in train_envs and regime_name in eval_envs:
            print(f"\nTraining subagent for {regime_name} regime...")
            
            # Create subagent
            subagent = SubAgent(
                env=train_envs[regime_name],
                model_name='maesac',  # Using SAC as the base algorithm
                regime_name=regime_name,
                policy="MlpPolicy",
                model_kwargs=config.SAC_PARAMS,
                seed=42
            )
            
            # Train subagent
            subagent.train(
                tb_log_name=f"subagent_{regime_name}",
                check_freq=1000,
                ck_dir=f"trained_models/hierarchical/subagents/{regime_name}",
                log_dir=f"results/hierarchical/subagents/{regime_name}",
                eval_env=eval_envs[regime_name],
                total_timesteps=50000  # Adjust based on available data
            )
            
            # Save trained subagent
            os.makedirs(f"trained_models/hierarchical/subagents/{regime_name}", exist_ok=True)
            subagent.save(f"trained_models/hierarchical/subagents/{regime_name}/model.zip")
            
            subagents[regime_name] = subagent
    
    return subagents

# Train hierarchical agent for portfolio management
def train_hierarchical_agent(full_train_env, full_eval_env, detector, subagents):
    """Train the hierarchical agent with meta-learning for portfolio management."""
    print("\nTraining hierarchical agent for portfolio management...")
    
    # Create hierarchical agent
    hierarchical_agent = HierarchicalAgent(
        env=full_train_env,
        regime_detector=detector,
        subagents=subagents,
        device=device
    )
    
    # Train meta-agent
    episode_rewards = hierarchical_agent.train_meta(
        n_episodes=100,  # Adjust based on available data
        batch_size=64,
        gamma=0.99
    )
    
    # Save hierarchical agent
    hierarchical_agent.save("trained_models/hierarchical")
    
    return hierarchical_agent, episode_rewards

# Evaluate agent performance
def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate agent performance on the given environment."""
    if env is None:
        return 0, 0
    
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    return mean_reward, std_reward

# Main function
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load stock data
    print(f"Loading stock data from {args.market}/{args.index}...")
    stock_data, tickers = load_stock_data(args.market, args.index)
    print(f"Loaded data for {len(stock_data)} stocks")
    
    # Create market index
    print("Creating market index from stock data...")
    index_data = create_market_index(stock_data, tickers)
    
    # Detect market regimes using index data
    print(f"Detecting market regimes using {args.method} with {args.n_regimes} regimes...")
    detector, train_index, eval_index, test_index, regime_boundaries = detect_regimes_with_index(
        index_data, n_regimes=args.n_regimes, method=args.method
    )
    
    # Get regime names
    regime_names = [detector.get_regime_name(i) for i in range(args.n_regimes)]
    print(f"Detected regimes: {regime_names}")
    
    # Label stock data with regimes
    print("Labeling stock data with detected regimes...")
    labeled_stock_data = label_stocks_with_regimes(
        stock_data, pd.concat([train_index, eval_index, test_index])
    )
    
    # Create regime-specific environments
    print("Creating regime-specific environments...")
    train_envs, eval_envs, test_envs, full_train_env, full_eval_env, full_test_env = create_regime_specific_envs(
        labeled_stock_data, tickers, detector, regime_names
    )
    
    # Train subagents
    print("Training subagents for each regime...")
    subagents = train_subagents(train_envs, eval_envs, regime_names)
    
    # Train hierarchical agent
    print("Training hierarchical agent for portfolio management...")
    hierarchical_agent, episode_rewards = train_hierarchical_agent(
        full_train_env, full_eval_env, detector, subagents
    )
    
    # Evaluate hierarchical agent
    print("\nEvaluating hierarchical agent...")
    mean_reward, std_reward = evaluate_agent(hierarchical_agent, full_test_env)
    print(f"Mean test reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Compare with baseline (single agent)
    print("\nTraining baseline agent...")
    baseline_agent = DRLAgent(full_train_env)
    baseline_model = baseline_agent.get_model(
        model_name='maesac',
        policy="MlpPolicy",
        model_kwargs=config.SAC_PARAMS,
        seed=42
    )
    
    # Train baseline model
    print("Training baseline model...")
    baseline_model = baseline_agent.train_model(
        model=baseline_model,
        tb_log_name="baseline_agent",
        check_freq=1000,
        ck_dir="trained_models/baseline",
        log_dir="results/baseline",
        eval_env=full_eval_env,
        total_timesteps=50000  # Same as subagents for fair comparison
    )
    
    # Save baseline model
    os.makedirs("trained_models/baseline", exist_ok=True)
    baseline_model.save("trained_models/baseline/model.zip")
    
    # Evaluate baseline agent
    print("\nEvaluating baseline agent...")
    baseline_mean_reward, baseline_std_reward = evaluate_agent(baseline_agent, full_test_env)
    print(f"Baseline mean test reward: {baseline_mean_reward:.2f} ± {baseline_std_reward:.2f}")
    
    # Compare results
    print("\nPerformance Comparison:")
    print(f"Hierarchical Agent: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Baseline Agent: {baseline_mean_reward:.2f} ± {baseline_std_reward:.2f}")
    
    improvement = ((mean_reward - baseline_mean_reward) / abs(baseline_mean_reward)) * 100 if baseline_mean_reward != 0 else 0
    print(f"Improvement: {improvement:.2f}%")
    
    # Save comparison results
    comparison_results = {
        'hierarchical_mean': mean_reward,
        'hierarchical_std': std_reward,
        'baseline_mean': baseline_mean_reward,
        'baseline_std': baseline_std_reward,
        'improvement_percentage': improvement,
        'regime_boundaries': regime_boundaries
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/comparison_results.pkl', 'wb') as f:
        pickle.dump(comparison_results, f)

if __name__ == "__main__":
    main()