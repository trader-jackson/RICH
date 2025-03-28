import yfinance as yf
import pandas as pd
import os

# Create directory for market index data
os.makedirs("market_index\\raw", exist_ok=True)

# Define market indices to download
indices = {
    '^GSPC': 'S&P500',  # S&P 500
    '^DJI': 'DJIA',    # Dow Jones Industrial Average
    '^IXIC': 'NASDAQ',  # NASDAQ Composite
    '^VIX': 'VIX'      # Volatility Index
}

# Download data for each index
for ticker, name in indices.items():
    print(f"Downloading {name} data...")
    
    # Download data from Yahoo Finance
    index_data = yf.download(
        ticker,
        start='2010-01-01',  # Adjust date range as needed
        end='2023-12-31',
        progress=False
    )
    if index_data.columns.nlevels != 1:
        index_data.columns = index_data.columns.droplevel(1)
    # Reset index to make date a column and flatten column names
    index_data = index_data.reset_index()
    index_data.columns = [col.lower() for col in index_data.columns]
    
    # Add index name and ticker columns
    index_data['index_name'] = name
    index_data['ticker'] = ticker
    
    # Save to CSV
    output_path = os.path.join("market_index",'raw', f'{name}.csv')
    index_data.to_csv(output_path, index=False)
    print(f"Saved {name} data to {output_path}")

print("\nMarket index data collection complete!")