import os
import numpy as np
import pandas as pd
import torch
import pickle
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

# Load and preprocess data
def load_data(data_path):
    """Load and preprocess data for training and testing."""
    df = pd.read_csv(data_path)
    
    # Ensure date is in datetime format
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Split data into train, validation, and test sets
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

# Detect market regimes
def detect_regimes(train_data, eval_data, test_data, n_regimes=4, method='hmm'):
    """Detect market regimes in the data."""
    # Create and fit regime detector on training data
    detector = RegimeDetector(n_regimes=n_regimes, method=method)
    train_data = detector.fit(train_data)
    
    # Predict regimes for evaluation and test data
    eval_data = detector.predict(eval_data)
    test_data = detector.predict(test_data)
    
    # Print regime statistics
    print("\nRegime Statistics (Training Data):")
    for regime in range(n_regimes):
        regime_name = detector.get_regime_name(regime)
        count = (train_data['regime'] == regime).sum()
        percentage = count / len(train_data) * 100
        print(f"  {regime_name}: {count} samples ({percentage:.2f}%)")
    
    return detector, train_data, eval_data, test_data

# Create environments for each regime
def create_environments(train_data, eval_data, test_data, detector, regime_names):
    """Create trading environments for each regime."""
    # Environment parameters
    env_params = {
        'stock_dim': 1,  # Assuming single stock for simplicity
        'hmax': 100,
        'initial_amount': 1000000,
        'transaction_cost_pct': 0.001,
        'reward_scaling': 1e-4,
        'state_space': 1,  # Add state_space parameter
        'action_space': 1,
        'tech_indicator_list': config.TECHICAL_INDICATORS,
        'temporal_feature_list': config.TEMPORAL_FEATURE,
        'additional_list': [],
        'time_window_start': [0],  # Starting index
        'episode_length': 252,  # One year of trading days
        'temporal_len': 60,
        'hidden_channel': 4,  # Add hidden_channel parameter
        'short_prediction_model_path': None,  # Add model paths
        'long_prediction_model_path': None,
        'make_plots': False,
        'print_verbosity': 0,
        'iteration': 0  # Add iteration parameter
    }
    
    # Create environments for each regime
    train_envs = {}
    eval_envs = {}
    test_envs = {}
    
    for regime_name in regime_names:
        # Create training environment
        train_env = create_regime_specific_env(
            StockTradingEnv,
            train_data,
            detector,
            regime_name,
            **env_params,
            mode='train'
        )
        train_envs[regime_name] = train_env
        
        # Create evaluation environment
        eval_env = create_regime_specific_env(
            StockTradingEnv,
            eval_data,
            detector,
            regime_name,
            **env_params,
            mode='eval'
        )
        eval_envs[regime_name] = eval_env
        
        # Create test environment
        test_env = create_regime_specific_env(
            StockTradingEnv,
            test_data,
            detector,
            regime_name,
            **env_params,
            mode='test'
        )
        test_envs[regime_name] = test_env
    
    # Create full environments for meta-agent training and testing
    full_train_env = StockTradingEnv(train_data, **env_params, mode='train')
    full_eval_env = StockTradingEnv(eval_data, **env_params, mode='eval')
    full_test_env = StockTradingEnv(test_data, **env_params, mode='test')
    
    return train_envs, eval_envs, test_envs, full_train_env, full_eval_env, full_test_env

# Train subagents
def train_subagents(train_envs, eval_envs, regime_names):
    """Train specialized subagents for each market regime."""
    subagents = {}
    
    for regime_name in regime_names:
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

# Train hierarchical agent
def train_hierarchical_agent(full_train_env, full_eval_env, detector, subagents):
    """Train the hierarchical agent with meta-learning."""
    print("\nTraining hierarchical agent...")
    
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
    # Load data
    data_path = "data/us_stocks/dow30/processed/AAPL.csv"  # Example path, adjust as needed
    train_data, eval_data, test_data = load_data(data_path)
    
    # Detect market regimes
    n_regimes = 4  # Bull, bear, volatile, stable
    detector, train_data, eval_data, test_data = detect_regimes(
        train_data, eval_data, test_data, n_regimes=n_regimes, method='hmm'
    )
    
    # Get regime names
    regime_names = [detector.get_regime_name(i) for i in range(n_regimes)]
    
    # Create environments
    train_envs, eval_envs, test_envs, full_train_env, full_eval_env, full_test_env = create_environments(
        train_data, eval_data, test_data, detector, regime_names
    )
    
    # Train subagents
    subagents = train_subagents(train_envs, eval_envs, regime_names)
    
    # Train hierarchical agent
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
    
    improvement = ((mean_reward - baseline_mean_reward) / abs(baseline_mean_reward)) * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Save comparison results
    comparison_results = {
        'hierarchical_mean': mean_reward,
        'hierarchical_std': std_reward,
        'baseline_mean': baseline_mean_reward,
        'baseline_std': baseline_std_reward,
        'improvement_percentage': improvement
    }
    
    with open('results/comparison_results.pkl', 'wb') as f:
        pickle.dump(comparison_results, f)