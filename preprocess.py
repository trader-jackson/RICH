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
        tech_indicator_list=config.TECHICAL_INDICATORS,
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
                      if col not in ['date', 'ticker'] + price_columns + diff_price_columns + ["price"]]
    
    processed_df = standardize_data(
        processed_df,
        feature_columns=feature_columns,
        price_columns=price_columns,
        diff_price_columns=diff_price_columns
    )
    
    return processed_df

def process_market_data(market_dir: str, save_dir: str = None) -> dict:
    """处理市场中的所有股票数据：
       1. 记录每个股票的交易天数；
       2. 过滤掉交易天数少于最高交易天数95%的股票；
       3. 对剩余股票进行特征工程，并将其日期索引对齐到交易天数最多的股票的日期，
          对缺失值进行前向填充；
       4. 将通过过滤的股票代码保存到一个文本文件中。

    参数:
        market_dir: str, 包含股票CSV文件的目录路径
        save_dir: str, 可选，处理后数据保存目录

    返回:
        包含处理后数据的字典
    """
    processed_data = {}
    trading_days = {}
    valid_tickers = []  # 记录通过过滤的股票

    # 确保保存目录存在
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 第一轮: 记录每个CSV文件的交易天数
    csv_files = [f for f in os.listdir(market_dir) if f.endswith('.csv')]
    for file in csv_files:
        file_path = os.path.join(market_dir, file)
        try:
            df = pd.read_csv(file_path)
            # 确保日期为datetime格式
            if isinstance(df['date'].iloc[0], str):
                df['date'] = pd.to_datetime(df['date'])
            # 以唯一日期计数
            trading_days[file] = df['date'].nunique()
        except Exception as e:
            print(f"读取 {file} 失败: {e}")
    
    if not trading_days:
        print("未找到有效的CSV文件。")
        return processed_data

    # 找到最高交易天数和对应的文件
    max_file = max(trading_days, key=trading_days.get)
    max_days = trading_days[max_file]
    threshold = 0.95 * max_days
    print(f"最高交易天数: {max_days}, 过滤阈值: {threshold}")

    # 获取主股票的日期索引（作为所有股票对齐的日期）
    master_df = pd.read_csv(os.path.join(market_dir, max_file))
    if isinstance(master_df['date'].iloc[0], str):
        master_df['date'] = pd.to_datetime(master_df['date'])
    # 排序并去重
    master_dates = pd.Series(master_df['date'].unique()).sort_values()
    master_dates = pd.DatetimeIndex(master_dates)

    # 第二轮: 处理满足条件的股票
    for file in csv_files:
        if trading_days[file] < threshold:
            print(f"过滤 {file}（交易天数: {trading_days[file]} < {threshold}）")
            continue
        
        ticker = file.split('.csv')[0]
        file_path = os.path.join(market_dir, file)
        
        try:
            df = pd.read_csv(file_path)
            processed_df = process_data(df, ticker=ticker)
            
            # 将日期设置为索引，确保日期为datetime格式
            if processed_df['date'].dtype == object:
                processed_df['date'] = pd.to_datetime(processed_df['date'])
            processed_df = processed_df.set_index('date').sort_index()
            
            # 对齐到 master_dates
            processed_df = processed_df.reindex(master_dates)
            
            # 填充缺失值：使用前向填充
            processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
            
            # 将 ticker 列恢复（如果需要，可重新添加）
            processed_df['ticker'] = ticker
            
            # 保存处理后的数据
            if save_dir:
                save_path = os.path.join(save_dir, f"{ticker}.csv")
                processed_df.reset_index().to_csv(save_path, index=False)
                print(f"处理后的 {ticker} 数据已保存到: {save_path}")
            
            # 存储结果
            processed_data[ticker] = processed_df.reset_index()  # index重置为日期列
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"处理 {file} 时发生错误: {e}")
    
    # 将有效的股票列表保存到文本文件
    if valid_tickers:
        if save_dir:
            parent_dir = os.path.dirname(save_dir)
            stock_list_path = os.path.join(parent_dir, "stock_list.txt")
        else:
            stock_list_path = os.path.join(market_dir, "stock_list.txt")
        with open(stock_list_path, 'w', encoding='utf-8') as f:
            for ticker in valid_tickers:
                f.write(ticker + "\n")
        print(f"有效股票列表已保存到: {stock_list_path}")
    
    return processed_data

def main():
    """主函数，用于处理所有市场的数据"""
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
