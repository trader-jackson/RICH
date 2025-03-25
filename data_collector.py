import pandas as pd
import tushare as ts
from typing import List, Dict, Union, Optional
from config import TRAIN_START_DATE,TEST_END_DATE
import yfinance as yf
import os
from tqdm import tqdm

class TushareCollector:
    """数据收集类，用于从不同数据源获取股票数据

    属性:
        ts_token: str
            Tushare API token
    """

    def __init__(self,token: str):
        self.ts_token = None
        self.set_tushare_token(token)

    def set_tushare_token(self, token: str):
        """设置Tushare API token"""
        self.ts_token = token
        ts.set_token(token)

    def get_trade_calendar(self, all_data: List[pd.DataFrame]) -> pd.DatetimeIndex:
        """从所有股票数据中获取交易日历

        参数:
            all_data: List[pd.DataFrame], 所有股票的数据列表

        返回:
            包含所有交易日期的DatetimeIndex
        """
        # 合并所有股票的交易日期
        all_dates = set()
        for df in all_data:
            if not df.empty:
                all_dates.update(df['date'].dt.date)
        
        # 转换为列表并排序
        trade_dates = sorted(list(all_dates))
        return pd.DatetimeIndex(trade_dates)

    

    def download_tushare_data(self, stock_list: List[str], start_date: str, end_date: str, max_retries: int = 3, retry_delay: int = 5) -> pd.DataFrame:
        """使用Tushare API下载A股数据并进行数据处理

        参数:
            stock_list: List[str], 股票代码列表
            start_date: str, 开始日期，格式：YYYYMMDD
            end_date: str, 结束日期，格式：YYYYMMDD
            max_retries: int, 最大重试次数
            retry_delay: int, 重试延迟（秒）

        返回:
            包含股票数据的DataFrame，经过以下处理：
            1. 筛选出交易天数超过80%的股票
            2. 对所有股票进行日期对齐
            3. 使用前值填充缺失数据
        """
        if not self.ts_token:
            raise ValueError("请先设置Tushare token")

        pro = ts.pro_api()
        all_data = []
        failed_stocks = []

        for stock_code in tqdm(stock_list, desc="下载股票数据"):
            attempt = 0
            while attempt < max_retries:
                try:
                    # 获取日线数据
                    df = pro.daily(
                        ts_code=stock_code,
                        start_date=start_date,
                        end_date=end_date,
                        timeout=30  # 增加超时时间到30秒
                    )
                    if not df.empty:
                        # 重命名列以匹配现有格式
                        df = df.rename(columns={
                            'trade_date': 'date',
                            'ts_code': 'ticker',
                            'vol': 'volume'
                        })
                        # 转换日期格式
                        df['date'] = pd.to_datetime(df['date'])
                        all_data.append(df)
                    break  # 成功获取数据，跳出重试循环
                except Exception as e:
                    attempt += 1
                    if attempt == max_retries:
                        print(f"下载{stock_code}数据失败，已重试{max_retries}次: {str(e)}")
                        failed_stocks.append(stock_code)
                        break
                    print(f"下载{stock_code}数据失败，正在进行第{attempt}次重试...")
                    import time
                    time.sleep(retry_delay)

        if all_data:
            # 获取交易日历
            trade_dates = self.get_trade_calendar(all_data)
            trade_dates = pd.to_datetime(trade_dates)
            
            # 合并所有数据
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.sort_values(['date', 'ticker'])
            
            # 计算每只股票的交易天数
            total_days = len(trade_dates)
            stock_days = final_df.groupby('ticker').size()
            active_stocks = stock_days[stock_days >= total_days * 0.8].index
            print("Number of active stocks:",len(active_stocks))
            # 筛选活跃股票
            final_df = final_df[final_df['ticker'].isin(active_stocks)]
            
            # 对每只股票进行重采样和填充
            stocks = []
            for ticker in active_stocks:
                stock_data = final_df[final_df['ticker'] == ticker].copy()
                # 创建完整的日期范围DataFrame
                date_range_df = pd.DataFrame({'date': trade_dates})
                # 合并数据，保持所有日期
                stock_data = pd.merge(date_range_df, stock_data, on='date', how='left')
                stock_data['ticker'] = ticker
                stock_data = stock_data.fillna(method='ffill')
                stocks.append(stock_data)
            
            # 合并所有处理后的数据
            final_df = pd.concat(stocks, ignore_index=True)
            final_df = final_df.sort_values(['date', 'ticker'])
            
            return final_df
        return pd.DataFrame()
    
    
class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None, auto_adjust=False, save_dir: str = None) -> pd.DataFrame:
        """Fetches data from Yahoo API and optionally saves individual stock data

        Parameters
        ----------
        proxy : str, optional
            proxy server URL
        auto_adjust : bool, optional
            adjust all OHLC automatically, by default False
        save_dir : str, optional
            directory to save individual stock CSV files

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all stock data
        """
        all_data = []
        num_failures = 0

        for ticker in tqdm(self.ticker_list, desc="下载股票数据"):
            try:
                # Download data for single stock
                temp_df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    proxy=proxy,
                    auto_adjust=auto_adjust,
                )
                
                if temp_df.empty:
                    print(f"警告：{ticker}没有数据")
                    num_failures += 1
                    continue

                if temp_df.columns.nlevels != 1:
                    temp_df.columns = temp_df.columns.droplevel(1)

                # Process single stock data
                temp_df = temp_df.reset_index()
                temp_df.rename(
                    columns={
                        "Date": "date",
                        "Adj Close": "price",
                        "Close": "close",
                        "High": "high",
                        "Low": "low",
                        "Volume": "volume",
                        "Open": "open",
                    },
                    inplace=True,
                )

               
                temp_df["ticker"] = ticker
                temp_df["day"] = temp_df["date"].dt.dayofweek
                temp_df["date"] = temp_df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
                
                # Save individual stock data if save_dir is provided
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    file_name = f"{ticker}.csv"
                    save_path = os.path.join(save_dir, file_name)
                    temp_df.to_csv(save_path, index=False)
                    print(f"已保存{ticker}数据到: {save_path}")

                all_data.append(temp_df)

            except Exception as e:
                print(f"下载{ticker}数据时出错: {str(e)}")
                num_failures += 1

        if num_failures == len(self.ticker_list):
            raise ValueError("没有成功下载任何数据")

        # Combine all stock data
        data_df = pd.concat(all_data, ignore_index=True)
        data_df = data_df.dropna()
        data_df = data_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
        print("数据集大小: ", data_df.shape)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.ticker.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["ticker", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.ticker.value_counts() >= mean_df)
        names = df.ticker.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.ticker.isin(select_stocks_list)]
        return df
    

class DataCollector:
    """沪深300数据收集器

    属性:
        tushare_collector: TushareCollector
            用于从Tushare获取数据的收集器
        yahoo_downloader: YahooDownloader
            用于从Yahoo Finance获取数据的下载器
    """

    def __init__(self, tushare_collector: TushareCollector):
        self.tushare_collector = tushare_collector
        self.yahoo_downloader = None

    def get_hs300_stocks(self, date: str) -> List[str]:
        """获取沪深300成分股列表，并转换为Yahoo Finance格式

        参数:
            date: str, 日期，格式：YYYYMMDD（仅用于兼容性，实际未使用）

        返回:
            成分股代码列表（Yahoo Finance格式）
        """
        from config_tickers import CSI_300_TICKER
        
        # 转换为Yahoo Finance格式
        yahoo_tickers = []
        for stock in CSI_300_TICKER:
            if stock.endswith('.SH'):
                yahoo_tickers.append(f"{stock[:-3]}.SS")
            elif stock.endswith('.SZ'):
                yahoo_tickers.append(f"{stock[:-3]}.SZ")
        return yahoo_tickers

    def download_hs_stocks(self, start_date: str, end_date: str, save_dir: str):
        """下载并保存沪深300成分股数据，每只股票保存为独立的CSV文件

        参数:
            start_date: str, 开始日期，格式：YYYYMMDD
            end_date: str, 结束日期，格式：YYYY-MM-DD
            save_dir: str, 保存数据的目录
        """
        # 获取沪深300成分股列表（Yahoo Finance格式）
        stock_list = self.get_hs300_stocks(end_date)
        print(f"获取到{len(stock_list)}只沪深300成分股")

        # 确保保存目录存在
        raw_dir = os.path.join(save_dir,"hs300", 'raw')
        os.makedirs(raw_dir, exist_ok=True)

        # 创建Yahoo下载器并下载每只股票的数据
        self.yahoo_downloader = YahooDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=stock_list
        )

        # 下载并保存每只股票的数据
        self.yahoo_downloader.fetch_data(save_dir=raw_dir)

    def download_us_stocks(self, start_date: str, end_date: str, save_dir: str):
        """下载并保存道琼斯30和纳斯达克100指数成分股数据，每只股票保存为独立的CSV文件

        参数:
            start_date: str, 开始日期，格式：YYYY-MM-DD
            end_date: str, 结束日期，格式：YYYY-MM-DD
            save_dir: str, 保存数据的目录
        """
        from config_tickers import DOW_30_TICKER, NAS_100_TICKER

        # 下载每个指数的成分股数据
        for index_name, ticker_list in {
            'dow30': DOW_30_TICKER,
            'nasdaq100': NAS_100_TICKER
        }.items():
            print(f"开始下载{index_name}指数成分股数据")
            
            # 确保保存目录存在
            index_save_dir = os.path.join(save_dir, index_name, 'raw')
            os.makedirs(index_save_dir, exist_ok=True)
            
            # 创建Yahoo下载器
            self.yahoo_downloader = YahooDownloader(
                start_date=start_date,
                end_date=end_date,
                ticker_list=ticker_list
            )
            
            # 下载并保存每只股票的数据
            self.yahoo_downloader.fetch_data(save_dir=index_save_dir)

def main():
    # 从private_config.py导入token
    from private_config import TUSHARE_TOKEN
    
    # 设置日期范围
    start_date = TRAIN_START_DATE # 训练开始日期
    end_date = TEST_END_DATE   # 当前日期
    
    # 设置保存目录
    base_save_dir = os.path.join('data')
    
    # 初始化下载器
    downloader = DataCollector(TUSHARE_TOKEN)
    
    # 下载沪深300数据
    hs300_save_dir = os.path.join(base_save_dir, 'cn_stocks')
    downloader.download_hs_stocks(start_date, end_date, hs300_save_dir)
    
    # 下载美股数据
    us_save_dir = os.path.join(base_save_dir, 'us_stocks')
    downloader.download_us_stocks(start_date, end_date, us_save_dir)

if __name__ == '__main__':
    main()