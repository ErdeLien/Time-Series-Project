import datetime as dt
import os
import zipfile
from multiprocessing import Pool

import polars as pl
import requests
from tqdm import tqdm

os.environ['TZ'] = 'UTC'  # Set the timezone for this script to UTC

# 扩充一下数据源，interest_rate,tick,options,futures
WORK_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORK_DIR, 'data')


class BinanceKlineDownloader:
    # Binance U Base Futures
    BASE_URL = 'https://data.binance.vision/data/futures/um/daily/klines'
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
               "taker_buy_quote_asset_volume", "ignore"]

    #['open_time','open','high','low','close','volume','close_time','quote_volume','count','taker_buy_volume','taker_buy_quote_volume','ignore']

    def __init__(self, symbol, interval, start_date, end_date, save_dir):
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.save_dir = os.path.join(save_dir, symbol)

        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def download_klines(self, date):
        year, month, day = date.year, date.month, date.day

        filename = f"{self.symbol}-{self.interval}-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.zip"
        csv_filename = filename.replace('.zip', '.csv')
        filepath = os.path.join(self.save_dir, filename)
        csv_filepath = os.path.join(self.save_dir, csv_filename)

        if os.path.exists(csv_filepath):
            # print(f"Data for {self.symbol} on {year}-{month}-{day} already exists. Skipping...")
            return csv_filepath

        url = f"{self.BASE_URL}/{self.symbol.upper()}/{self.interval}/{filename}"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.save_dir)

            # Check if CSV has headers
            with open(csv_filepath, 'r') as f:
                first_line = f.readline().strip()
                has_header = not any(char.isdigit() for char in first_line)  # Check if there's any number in the first line

            df = pl.read_csv(csv_filepath, has_header=has_header, new_columns=self.columns)
            df = df.with_columns(
                pl.col('open_time') - int(date.timestamp() * 1000)
            )
            df.write_csv(csv_filepath)
            os.remove(filepath)

            return csv_filepath

        except requests.HTTPError as e:
            print(f'Failed to fetch data for {self.symbol} on {year}-{month}-{day}. HTTP error: {e}')
        except Exception as e:
            print(f"An error occurred while processing {self.symbol} on {year}-{month}-{day}. Error: {e}")

    def worker(self, date):
        return self.download_klines(date)

    def download(self, core):
        date_list = [self.start_date + dt.timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]

        # Apply the progress bar using tqdm
        with Pool(core) as p:
            for _ in tqdm(p.imap(self.worker, date_list), total=len(date_list)):
                pass


class BinanceTradesDownloader:
    BASE_URL = 'https://data.binance.vision/data/futures/um/daily/trades'
    columns = ["trade_id", "price", "qty", "quoteQty", "time", "isBuyerMaker"]

    def __init__(self, symbol, start_date, end_date, save_dir):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.save_dir = os.path.join(save_dir, symbol)

        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def download_trades(self, date):
        year, month, day = date.year, date.month, date.day

        filename = f"{self.symbol}-trades-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.zip"
        csv_filename = filename.replace('.zip', '.csv')
        filepath = os.path.join(self.save_dir, filename)
        csv_filepath = os.path.join(self.save_dir, csv_filename)

        if os.path.exists(csv_filepath):
            return csv_filepath

        url = f"{self.BASE_URL}/{self.symbol.upper()}/{filename}"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.save_dir)

            # Check if CSV has headers
            with open(csv_filepath, 'r') as f:
                first_line = f.readline().strip()
                has_header = not any(
                    char.isdigit() for char in first_line)  # Check if there's any number in the first line

            df = pl.read_csv(csv_filepath, has_header=has_header, new_columns=self.columns)

            df = df.with_columns(
                pl.col('time') - int(date.timestamp() * 1000)
            )
            df.write_csv(csv_filepath)
            os.remove(filepath)

            return csv_filepath

        except requests.HTTPError as e:
            print(f'Failed to fetch data for {self.symbol} on {year}-{month}-{day}. HTTP error: {e}')
        except Exception as e:
            print(f"An error occurred while processing {self.symbol} on {year}-{month}-{day}. Error: {e}")

    def worker(self, date):
        return self.download_trades(date)

    def download(self, core):
        date_list = [self.start_date + dt.timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]

        with Pool(core) as p:
            for _ in tqdm(p.imap(self.worker, date_list), total=len(date_list)):
                pass

if __name__ == "__main__":
    core = 7
    symbol = 'ETHUSDT'
    interval = '1m'
    start_date = dt.datetime(2019, 9, 12)
    end_date = dt.datetime(2023, 9, 1)

    traders_downloader = BinanceTradesDownloader(symbol, start_date, end_date, os.path.join(DATA_DIR, 'trades'))
    traders_downloader.download(core)

