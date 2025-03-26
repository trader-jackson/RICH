# MultiRL-Portfolio

A multi-agent reinforcement learning framework for financial portfolio management. This project implements various RL algorithms to optimize stock trading strategies using historical market data and technical indicators.

## Overview

This project combines reinforcement learning with transformer-based forecasting models to create an intelligent portfolio management system. The framework supports:

- Multiple RL algorithms for portfolio optimization
- Technical indicator-based feature engineering
- Transformer models for price prediction
- Customizable trading environments
- Support for both Chinese and US stock markets

## Project Structure

```
├── Transformer/               # Transformer model implementation
│   ├── attn.py               # Attention mechanisms
│   ├── embed.py              # Embedding layers
│   ├── layer.py              # Transformer layers
│   ├── transformer.py        # Main transformer model
│   └── transformer_layer.py  # Transformer layer implementation
├── data/                     # Stock market data
│   ├── cn_stocks/            # Chinese stock market data
│   │   └── hs300/            # CSI 300 Index stocks
│   └── us_stocks/            # US stock market data
│       ├── dow30/            # Dow Jones 30 stocks
│       └── nasdaq100/        # NASDAQ 100 stocks
├── config.py                 # Configuration settings
├── config_tickers.py         # Stock ticker configurations
├── data_collector.py         # Data collection utilities
├── env_stocktrading_forecasting.py  # RL environment for stock trading
├── feature_engineering.py    # Feature engineering for stock data
├── preprocess.py             # Data preprocessing utilities
└── private_config.py         # Private configuration (API keys, etc.)
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Stable-Baselines3
- Pandas, NumPy, Matplotlib
- Tushare (for Chinese stock data)
- yfinance (for US stock data)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MultiRL-Portfolio.git
   cd MultiRL-Portfolio
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API tokens:
   - Edit `private_config.py` to add your Tushare API token:
     ```python
     TUSHARE_TOKEN="your_tushare_token_here"
     ```

## Usage

### Data Collection

To collect stock data from Tushare (Chinese stocks):

```python
from data_collector import TushareCollector
from private_config import TUSHARE_TOKEN
from config_tickers import CSI_300_TICKER

# Initialize collector
collector = TushareCollector(token=TUSHARE_TOKEN)

# Download data for CSI 300 stocks
data = collector.download_tushare_data(
    stock_list=CSI_300_TICKER[:10],  # First 10 stocks
    start_date="2020-01-01",
    end_date="2021-12-31"
)
```

For US stocks using yfinance:

```python
from data_collector import download_yfinance_data

# Download data for US stocks
data = download_yfinance_data(
    stock_list=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2021-12-31"
)
```

### Feature Engineering

```python
from feature_engineering import FeatureEngineer
from config import INDICATORS

# Initialize feature engineer
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS
)

# Process data
processed_data = fe.preprocess_data(data)
```

### Training RL Agents

The project supports various RL algorithms from Stable-Baselines3, including PPO, A2C, DDPG, TD3, and SAC.

```python
from stable_baselines3 import PPO
from env_stocktrading_forecasting import StockTradingEnv

# Create environment
env = StockTradingEnv(
    df=processed_data,
    stock_dim=len(stock_list),
    hmax=100,  # Maximum number of shares to trade
    initial_amount=1000000,  # Initial capital
    transaction_cost_pct=0.001,  # Transaction cost
    reward_scaling=1e-4,
    action_space=len(stock_list),
    tech_indicator_list=INDICATORS,
    temporal_feature_list=["open", "high", "low", "close", "volume"],
    additional_list=[],
    time_window_start=[0],
    mode="train"
)

# Train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("ppo_portfolio_model")
```

## Configuration

The project uses several configuration files:

- `config.py`: Contains general settings like date ranges, model parameters, and technical indicators
- `config_tickers.py`: Lists stock tickers for different markets
- `private_config.py`: Stores private API keys and tokens

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [Tushare](https://tushare.pro/) for Chinese market data
- [yfinance](https://github.com/ranaroussi/yfinance) for US market data
