# Hierarchical Reinforcement Learning for Portfolio Management

This README explains how to use the hierarchical reinforcement learning agent for portfolio management using your stock data.

## Data Structure

The system expects data to be organized in the following structure:
```
data/
  us_stocks/
    dow30/
      processed/
        AAPL.csv
        MSFT.csv
        ...
    nasdaq100/
      processed/
        AAPL.csv
        GOOGL.csv
        ...
  cn_stocks/
    hs300/
      processed/
        ...
```

Each CSV file should contain preprocessed stock data with features like open, close, high, low, volume, and technical indicators.

## Running the Hierarchical Agent

You can run the hierarchical agent using the `train_hierarchical_with_data.py` script. This script allows you to specify which market, index, and stock ticker to use for training.

### Command Line Arguments

- `--market`: Market to use (us_stocks, cn_stocks). Default: us_stocks
- `--index`: Index to use (dow30, nasdaq100, hs300). Default: dow30
- `--ticker`: Stock ticker to train on. If not specified, the first available ticker will be used.
- `--n_regimes`: Number of market regimes to detect. Default: 4
- `--method`: Method for regime detection (hmm, kmeans). Default: hmm

### Example Usage

```bash
# Train on AAPL from Dow 30 index
python train_hierarchical_with_data.py --ticker AAPL

# Train on GOOGL from NASDAQ 100 index
python train_hierarchical_with_data.py --market us_stocks --index nasdaq100 --ticker GOOGL

# Train with 3 market regimes using kmeans clustering
python train_hierarchical_with_data.py --ticker MSFT --n_regimes 3 --method kmeans
```

## Training Process

The training process follows these steps:

1. **Data Loading**: Loads the specified stock data and splits it into training, evaluation, and test sets based on the date ranges in config.py.

2. **Regime Detection**: Detects market regimes in the training data using either Hidden Markov Models (HMM) or K-means clustering.

3. **Subagent Training**: Trains specialized reinforcement learning agents for each detected market regime.

4. **Meta-agent Training**: Trains a meta-agent that learns to combine the actions of the specialized subagents.

5. **Evaluation**: Evaluates the hierarchical agent against a baseline single-agent approach.

## Results

After training, the results will be saved in the following directories:

- Trained models: `trained_models/hierarchical/`
- Performance logs: `results/hierarchical/`
- TensorBoard logs: `tensorboard_log/hierarchical/`

A comparison between the hierarchical agent and the baseline agent will be saved as a pickle file in `results/{market}/{index}/{ticker}_comparison_results.pkl`.