import numpy as np
import pandas as pd
import tushare as ts
from stockstats import StockDataFrame as Sdf
from typing import List, Dict, Union, Optional
import config
import warnings

# 设置警告过滤
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class FeatureEngineer:
    """特征工程类，用于处理股票数据并计算技术指标

    属性:
        use_technical_indicator: bool
            是否使用技术指标
        tech_indicator_list: list
            技术指标列表
        use_vix: bool
            是否使用VIX指数
        use_turbulence: bool
            是否使用波动指数
        user_defined_feature: bool
            是否使用用户自定义特征
    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.TECHICAL_INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature




    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据，添加技术指标

        参数:
            df: pd.DataFrame, 包含OHLCV数据的DataFrame

        返回:
            处理后的DataFrame
        """
        # 清洗数据
        df = self.clean_data(df)

        # 添加技术指标
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("成功添加技术指标")

        # 添加用户自定义特征
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("成功添加用户自定义特征")

        # 填充缺失值
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗数据

        参数:
            data: pd.DataFrame, 原始数据

        返回:
            清洗后的DataFrame
        """
        df = data.copy()
        df = df.sort_values(["date", "ticker"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="ticker", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.ticker.isin(tics)]
        return df

    def add_technical_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标

        参数:
            data: pd.DataFrame, 原始数据

        返回:
            添加技术指标后的DataFrame
        """
        df = data.copy()
        df = df.sort_values(by=["ticker", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.ticker.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.ticker == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["ticker"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.ticker == unique_ticker[i]]["date"].to_list()
                    indicator_df = pd.concat([indicator_df, temp_indicator], ignore_index=True)
                except Exception as e:
                    if isinstance(e, (RuntimeWarning, FutureWarning)):
                        continue
                    warnings.warn(f"计算{indicator}指标时出错: {e}", UserWarning)
            df = df.merge(indicator_df[["ticker", "date", indicator]], on=["ticker", "date"], how="left")

        df = df.sort_values(by=["date", "ticker"])
        return df

    def add_user_defined_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加用户自定义特征

        参数:
            data: pd.DataFrame, 原始数据

        返回:
            添加用户自定义特征后的DataFrame
        """
        df = data.copy()
       
        
        # 计算OHLC的差分特征
        df["dopen"] = df.groupby("ticker")["open"].diff()
        df["dhigh"] = df.groupby("ticker")["high"].diff()
        df["dlow"] = df.groupby("ticker")["low"].diff()
        df["dclose"] = df.groupby("ticker")["close"].diff()
        df["dvolume"] = df.groupby("ticker")["volume"].diff()
        return df
    

        

def standardize_data(df: pd.DataFrame, feature_columns: List[str], price_columns: List[str] = None, diff_price_columns: List[str] = None) -> pd.DataFrame:
    """标准化数据
    
    参数:
        df: pd.DataFrame, 原始数据
        feature_columns: List[str], 需要标准化的特征列表
        price_columns: List[str], 价格列(open, high, low, close)，将按照每只股票分别除以其最大值
        diff_price_columns: List[str], 差分价格列(dopen, dhigh, dlow, dclose)，将按照每只股票分别除以其最大值
    
    返回:
        标准化后的DataFrame
    """
    df_copy = df.copy()
    
    # 对价格列进行最大值标准化（按股票分组）
    if price_columns:
        for ticker in df_copy['ticker'].unique():
            ticker_mask = df_copy['ticker'] == ticker
            # 找出该股票价格列中的最大值
            price_max = df_copy.loc[ticker_mask, price_columns].max().max()
            # 对该股票的每个价格列进行标准化
            for col in price_columns:
                df_copy.loc[ticker_mask, col] = df_copy.loc[ticker_mask, col] / price_max
    
    # 对差分价格列进行最大值标准化（按股票分组）
    if diff_price_columns:
        for ticker in df_copy['ticker'].unique():
            ticker_mask = df_copy['ticker'] == ticker
            # 找出该股票差分价格列中的最大值
            diff_price_max = df_copy.loc[ticker_mask, diff_price_columns].max().max()
            # 对该股票的每个差分价格列进行标准化
            for col in diff_price_columns:
                df_copy.loc[ticker_mask, col] = df_copy.loc[ticker_mask, col] / diff_price_max
    
    # 对其他特征列使用均值和标准差进行标准化（按股票分组）
    for col in feature_columns:
        if (price_columns and col in price_columns) or (diff_price_columns and col in diff_price_columns):
            continue
        for ticker in df_copy['ticker'].unique():
            ticker_mask = df_copy['ticker'] == ticker
            mean = df_copy.loc[ticker_mask, col].mean()
            std = df_copy.loc[ticker_mask, col].std()
            df_copy.loc[ticker_mask, col] = (df_copy.loc[ticker_mask, col] - mean) / std
    return df_copy