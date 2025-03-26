import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.noise import NormalActionNoise
import os
import pickle

from regime_detector import RegimeDetector
from DRLAgent import DRLAgent, MODELS


class MetaAgent(nn.Module):
    """
    Meta-agent that learns to combine actions from specialized subagents.
    
    This neural network takes the current state observation and outputs weights
    for each subagent. These weights are used to combine the actions of subagents
    into a final action.
    """
    
    def __init__(self, state_dim, n_subagents, hidden_dim=64):
        """
        Initialize the meta-agent.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state observation
        n_subagents : int
            Number of subagents to combine
        hidden_dim : int, default=64
            Dimension of hidden layers
        """
        super(MetaAgent, self).__init__()
        
        self.state_dim = state_dim
        self.n_subagents = n_subagents
        
        # Neural network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_subagents)
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        state : torch.Tensor
            The current state observation
            
        Returns:
        --------
        torch.Tensor
            Weights for each subagent (softmax applied)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply softmax to ensure weights are non-negative and sum to 1
        weights = F.softmax(x, dim=-1)
        
        return weights


class SubAgent:
    """
    Wrapper for a specialized DRL agent trained on a specific market regime.
    """
    
    def __init__(self, env, model_name, regime_name, policy="MlpPolicy", 
                 policy_kwargs=None, model_kwargs=None, verbose=1, seed=None):
        """
        Initialize a subagent.
        
        Parameters:
        -----------
        env : gym.Env
            The trading environment
        model_name : str
            Name of the DRL model to use
        regime_name : str
            Name of the market regime this agent specializes in
        policy : str, default="MlpPolicy"
            Policy type to use
        policy_kwargs : dict, optional
            Additional arguments for the policy
        model_kwargs : dict, optional
            Additional arguments for the model
        verbose : int, default=1
            Verbosity level
        seed : int, optional
            Random seed
        """
        self.env = env
        self.model_name = model_name
        self.regime_name = regime_name
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.model_kwargs = model_kwargs
        self.verbose = verbose
        self.seed = seed
        
        # Create DRL agent
        self.drl_agent = DRLAgent(env)
        self.model = None
        
    def train(self, tb_log_name, check_freq, ck_dir, log_dir, eval_env, total_timesteps=5000):
        """
        Train the subagent on data from its specialized regime.
        
        Parameters:
        -----------
        tb_log_name : str
            Name for tensorboard logs
        check_freq : int
            Frequency for saving checkpoints
        ck_dir : str
            Directory for saving checkpoints
        log_dir : str
            Directory for saving logs
        eval_env : gym.Env
            Environment for evaluation during training
        total_timesteps : int, default=5000
            Total number of timesteps to train for
            
        Returns:
        --------
        self
            The trained subagent
        """
        # Create model if it doesn't exist
        if self.model is None:
            self.model = self.drl_agent.get_model(
                model_name=self.model_name,
                policy=self.policy,
                policy_kwargs=self.policy_kwargs,
                model_kwargs=self.model_kwargs,
                verbose=self.verbose,
                seed=self.seed
            )
        
        # Train the model
        self.model = self.drl_agent.train_model(
            model=self.model,
            tb_log_name=f"{tb_log_name}_{self.regime_name}",
            check_freq=check_freq,
            ck_dir=os.path.join(ck_dir, self.regime_name),
            log_dir=os.path.join(log_dir, self.regime_name),
            eval_env=eval_env,
            total_timesteps=total_timesteps
        )
        
        return self
    
    def load(self, model_path):
        """
        Load a pre-trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
            
        Returns:
        --------
        self
            The subagent with loaded model
        """
        if self.model_name not in MODELS:
            raise NotImplementedError(f"Model {self.model_name} not implemented")
        
        self.model = MODELS[self.model_name].load(model_path)
        return self
    
    def save(self, save_path):
        """
        Save the trained model.
        
        Parameters:
        -----------
        save_path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        self.model.save(save_path)
    
    def act(self, state):
        """
        Get an action from the subagent for the given state.
        
        Parameters:
        -----------
        state : numpy.ndarray
            The current state observation
            
        Returns:
        --------
        numpy.ndarray
            The action to take
        """
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        action, _ = self.model.predict(state, deterministic=True)
        return action


class HierarchicalAgent:
    """
    Hierarchical agent that combines specialized subagents using a meta-agent.
    
    This agent detects the current market regime, gets actions from specialized
    subagents, and combines them using weights from the meta-agent.
    """
    
    def __init__(self, env, regime_detector, subagents, meta_agent=None, device='cpu'):
        """
        Initialize the hierarchical agent.
        
        Parameters:
        -----------
        env : gym.Env
            The trading environment
        regime_detector : RegimeDetector
            Detector for identifying market regimes
        subagents : dict
            Dictionary mapping regime names to SubAgent instances
        meta_agent : MetaAgent, optional
            Meta-agent for combining subagent actions
        device : str, default='cpu'
            Device to run the meta-agent on ('cpu' or 'cuda')
        """
        self.env = env
        self.regime_detector = regime_detector
        self.subagents = subagents
        self.meta_agent = meta_agent
        self.device = device
        
        # Get state dimension from environment
        self.state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        
        # Create meta-agent if not provided
        if meta_agent is None:
            self.meta_agent = MetaAgent(
                state_dim=self.state_dim,
                n_subagents=len(subagents),
                hidden_dim=64
            )
        
        # Move meta-agent to device
        self.meta_agent.to(device)
        
        # Optimizer for meta-agent
        self.optimizer = optim.Adam(self.meta_agent.parameters(), lr=0.001)
        
        # For tracking performance
        self.regime_counts = {regime: 0 for regime in subagents.keys()}
        self.weights_history = []
    
    def act(self, state):
        """
        Get a combined action from subagents for the given state.
        
        Parameters:
        -----------
        state : numpy.ndarray
            The current state observation
            
        Returns:
        --------
        numpy.ndarray
            The combined action to take
        """
        # Flatten state for meta-agent
        flat_state = state.flatten()
        state_tensor = torch.FloatTensor(flat_state).to(self.device)
        
        # Get weights from meta-agent
        with torch.no_grad():
            weights = self.meta_agent(state_tensor).cpu().numpy()
        
        # Get actions from each subagent
        actions = []
        for subagent in self.subagents.values():
            action = subagent.act(state)
            actions.append(action)
        
        # Combine actions using weights
        actions = np.array(actions)
        combined_action = np.sum(actions * weights.reshape(-1, 1), axis=0)
        
        # Store weights for analysis
        self.weights_history.append(weights)
        
        return combined_action
    
    def train_meta(self, n_episodes=100, batch_size=64, gamma=0.99):
        """
        Train the meta-agent using policy gradient.
        
        Parameters:
        -----------
        n_episodes : int, default=100
            Number of episodes to train for
        batch_size : int, default=64
            Batch size for training
        gamma : float, default=0.99
            Discount factor
            
        Returns:
        --------
        list
            Episode rewards during training
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            log_probs = []
            rewards = []
            
            
            while not done:
                # Flatten state for meta-agent
                flat_state = state.flatten()
                state_tensor = torch.FloatTensor(flat_state).to(self.device)
                
                # Get weights from meta-agent
                weights = self.meta_agent(state_tensor)
                
                # Get actions from each subagent
                actions = []
                for subagent in self.subagents.values():
                    action = subagent.act(state)
                    actions.append(action)
                
                # Combine actions using weights
                actions = np.array(actions)
                combined_action = np.sum(actions * weights.cpu().detach().numpy().reshape(-1, 1), axis=0)
                
                # Take action in environment
                next_state, reward, done, _ = self.env.step(combined_action)
                
                # Store log probability and reward
                log_prob = torch.log(weights + 1e-10).sum()
                log_probs.append(log_prob)
                rewards.append(reward)
                
                # Update state
                state = next_state
                episode_reward += reward
            
            # Calculate returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)
            
            # Calculate loss
            loss = []
            for log_prob, R in zip(log_probs, returns):
                loss.append(-log_prob * R)
            
            # Optimize meta-agent
            self.optimizer.zero_grad()
            loss = torch.stack(loss).sum()
            loss.backward()
            self.optimizer.step()
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}, Reward: {episode_reward:.2f}")
        
        return episode_rewards
    
    def save(self, save_dir):
        """
        Save the hierarchical agent.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save the agent
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save meta-agent
        torch.save(self.meta_agent.state_dict(), os.path.join(save_dir, 'meta_agent.pth'))
        
        # Save subagents
        for regime_name, subagent in self.subagents.items():
            subagent_dir = os.path.join(save_dir, regime_name)
            os.makedirs(subagent_dir, exist_ok=True)
            subagent.save(os.path.join(subagent_dir, 'model.zip'))
        
        # Save regime detector
        self.regime_detector.save(os.path.join(save_dir, 'regime_detector.pkl'))
        
        # Save weights history
        with open(os.path.join(save_dir, 'weights_history.pkl'), 'wb') as f:
            pickle.dump(self.weights_history, f)
    
    @classmethod
    def load(cls, env, load_dir, device='cpu'):
        """
        Load a hierarchical agent.
        
        Parameters:
        -----------
        env : gym.Env
            The trading environment
        load_dir : str
            Directory to load the agent from
        device : str, default='cpu'
            Device to run the meta-agent on ('cpu' or 'cuda')
            
        Returns:
        --------
        HierarchicalAgent
            The loaded hierarchical agent
        """
        # Load regime detector
        regime_detector = RegimeDetector.load(os.path.join(load_dir, 'regime_detector.pkl'))
        
        # Get regime names
        regime_names = [d for d in os.listdir(load_dir) 
                       if os.path.isdir(os.path.join(load_dir, d)) 
                       and d != 'regime_detector']
        
        # Load subagents
        subagents = {}
        for regime_name in regime_names:
            subagent = SubAgent(env, model_name='maesac', regime_name=regime_name)
            subagent.load(os.path.join(load_dir, regime_name, 'model.zip'))
            subagents[regime_name] = subagent
        
        # Create meta-agent
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        meta_agent = MetaAgent(state_dim=state_dim, n_subagents=len(subagents))
        meta_agent.load_state_dict(torch.load(os.path.join(load_dir, 'meta_agent.pth')))
        
        # Create hierarchical agent
        hierarchical_agent = cls(env, regime_detector, subagents, meta_agent, device)
        
        # Load weights history if available
        weights_history_path = os.path.join(load_dir, 'weights_history.pkl')
        if os.path.exists(weights_history_path):
            with open(weights_history_path, 'rb') as f:
                hierarchical_agent.weights_history = pickle.load(f)
        
        return hierarchical_agent


def filter_data_by_regime(df, regime_detector, regime_name):
    """
    Filter data to include only samples from a specific regime.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data and other features
    regime_detector : RegimeDetector
        Detector for identifying market regimes
    regime_name : str
        Name of the regime to filter for
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only data from the specified regime
    """
    # Detect regimes if not already done
    if 'regime' not in df.columns:
        df = regime_detector.fit(df)
    
    # Get numerical regime ID for the given name
    regime_id = None
    for rid, rname in regime_detector.regime_mapping.items():
        if rname == regime_name:
            regime_id = rid
            break
    
    if regime_id is None:
        raise ValueError(f"Regime '{regime_name}' not found in detector mapping")
    
    # Filter data
    filtered_df = df[df['regime'] == regime_id].copy()
    
    return filtered_df


def create_regime_specific_env(env_class, df, regime_detector, regime_name, **env_kwargs):
    """
    Create an environment with data filtered for a specific regime.
    
    Parameters:
    -----------
    env_class : class
        Environment class to instantiate
    df : pandas.DataFrame
        DataFrame containing price data and other features
    regime_detector : RegimeDetector
        Detector for identifying market regimes
    regime_name : str
        Name of the regime to filter for
    **env_kwargs : dict
        Additional arguments for the environment
        
    Returns:
    --------
    gym.Env
        Environment with data filtered for the specified regime
    """
    # Filter data for the specific regime
    filtered_df = filter_data_by_regime(df, regime_detector, regime_name)
    
    # Create environment with filtered data
    env = env_class(filtered_df, **env_kwargs)
    
    return env