import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer, standardize_data
import config
import os

def process_data(df: pd.DataFrame, ticker: str = None, save_path: str = None) -> pd.DataFrame:
    """处理原始数据，生成技术指标和差分特征

    参数:
        df: pd.DataFrame, 原始数据DataFrame
        ticker: str, 可选，股票代码
        save_path: str, 可选，处理后数据保存路径

    返回:
        处理后的DataFrame
    """
    # 确保日期列格式正确
    if isinstance(df['date'].iloc[0], str):
        df['date'] = pd.to_datetime(df['date'])
    
    # 添加ticker列（如果不存在）
    if 'ticker' not in df.columns and ticker is not None:
        df['ticker'] = ticker
    
    # 初始化特征工程类
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=True
    )
    
    # 处理数据
    processed_df = fe.preprocess_data(df)
    
    # 标准化数据
    price_columns = ['open', 'high', 'low', 'close']
    diff_price_columns = ['dopen', 'dhigh', 'dlow', 'dclose']
    feature_columns = [col for col in processed_df.columns 
                      if col not in ['date', 'ticker'] + price_columns + diff_price_columns+["price"]]
    
    processed_df = standardize_data(
        processed_df,
        feature_columns=feature_columns,
        price_columns=price_columns,
        diff_price_columns=diff_price_columns
    )
    
    # 保存处理后的数据
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        processed_df.to_csv(save_path, index=False)
        print(f'处理后的数据已保存到: {save_path}')
    
    return processed_df

def process_market_data(market_dir: str, save_dir: str = None) -> dict:
    """处理市场中的所有股票数据

    参数:
        market_dir: str, 包含股票CSV文件的目录路径
        save_dir: str, 可选，处理后数据保存目录

    返回:
        包含处理后数据的字典
    """
    processed_data = {}
    
    # 确保保存目录存在
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 处理目录中的每个CSV文件
    for file in os.listdir(market_dir):
        if file.endswith('.csv'):
            ticker = file.split('.csv')[0]  # 假设文件名格式为: TICKER_*.csv
            file_path = os.path.join(market_dir, file)
            
            # 读取并处理数据
            df = pd.read_csv(file_path)
            processed_df = process_data(
                df,
                ticker=ticker,
                save_path=os.path.join(save_dir, f"processed_{ticker}.csv") if save_dir else None
            )
            processed_data[ticker] = processed_df
    
    return processed_data

def main():
    """主函数，用于处理所有市场的数据"""
    import os
    import pandas as pd
    
    # 设置数据目录
    base_data_dir = os.path.join('data')
    
    # 处理沪深300数据
    hs300_raw_dir = os.path.join(base_data_dir, 'cn_stocks', 'hs300', 'raw')
    hs300_processed_dir = os.path.join(base_data_dir, 'cn_stocks', 'hs300', 'preprocessed')
    
    if os.path.exists(hs300_raw_dir):
        print('处理沪深300数据...')
        process_market_data(hs300_raw_dir, save_dir=hs300_processed_dir)
    
    # 处理道琼斯30数据
    dow30_raw_dir = os.path.join(base_data_dir, 'us_stocks', 'dow30', 'raw')
    dow30_processed_dir = os.path.join(base_data_dir, 'us_stocks', 'dow30', 'processed')
    
    if os.path.exists(dow30_raw_dir):
        print('处理道琼斯30数据...')
        process_market_data(dow30_raw_dir, save_dir=dow30_processed_dir)
    
    # 处理纳斯达克100数据
    nasdaq100_raw_dir = os.path.join(base_data_dir, 'us_stocks', 'nasdaq100', 'raw')
    nasdaq100_processed_dir = os.path.join(base_data_dir, 'us_stocks', 'nasdaq100', 'processed')
    
    if os.path.exists(nasdaq100_raw_dir):
        print('处理纳斯达克100数据...')
        process_market_data(nasdaq100_raw_dir, save_dir=nasdaq100_processed_dir)

if __name__ == '__main__':
    main()