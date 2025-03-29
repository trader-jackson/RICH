import yfinance as yf
import pandas as pd
import os
import tushare as ts
TUSHARE_TOKEN="d0205b6fb350e5c87364c36fbbd8cae2263ee1acdce56eb5a3a442ff"

# Create directory for market index data
os.makedirs(os.path.join("market_index", "raw"), exist_ok=True)

# Define market indices to download: Dow30, Nasdaq100
indices = {
    '^DJI': 'Dow30',        # Dow 30 index (Dow Jones Industrial Average)
    '^NDX': 'Nasdaq100',     # Nasdaq 100 index
}

# Initialize Tushare API
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# Download SSE50 index data using Tushare
print("Downloading SSE50 data...")
sse50_data = pro.index_daily(
    ts_code='000016.SH',
    start_date='20100101',
    end_date='20241231'
)

# Process SSE50 data
if not sse50_data.empty:
    sse50_data = sse50_data.rename(columns={
        'trade_date': 'date',
        'ts_code': 'ticker',
        'vol': 'volume'
    })
    sse50_data['date'] = pd.to_datetime(sse50_data['date'])
    sse50_data['index_name'] = 'SSE50'
    
    # Save SSE50 data
    output_path = os.path.join("market_index", "raw", 'SSE50.csv')
    sse50_data.sort_values(by='date', inplace=True)
    sse50_data.to_csv(output_path, index=False)
    print(f"Saved SSE50 data to {output_path}")

# Download data for other indices using Yahoo Finance
for ticker, name in indices.items():
    print(f"Downloading {name} data...")
    
    # Download data from Yahoo Finance
    index_data = yf.download(
        ticker,
        start='2010-01-01',  # Adjust date range as needed
        end='2024-12-31',
        progress=False
    )
    
    # If the data has multi-level columns, drop the extra level
    if index_data.columns.nlevels != 1:
        index_data.columns = index_data.columns.droplevel(1)
    
    # Reset index to make date a column and flatten column names
    index_data = index_data.reset_index()
    index_data.columns = [col.lower() for col in index_data.columns]
    
    # Add index name and ticker columns
    index_data['index_name'] = name
    index_data['ticker'] = ticker
    
    # Save to CSV
    output_path = os.path.join("market_index", "raw", f'{name}.csv')
    index_data.to_csv(output_path, index=False)
    print(f"Saved {name} data to {output_path}")

print("\nMarket index data collection complete!")
