import sys
import os
sys.path.append(os.path.abspath("C:/Users/Admin/Desktop/git/crypto/"))
from cache_maker import cache_maker
import os
import pandas as pd
import datetime
import numpy as np
import polars as pl
    
class Low(cache_maker):
    def __init__(self,interval,memmap_name,start,end,crypto,data):
        super().__init__(interval,memmap_name,start,end,crypto)
        self.data = data


    def calculate(self,tidx,crypto,year,month,day):
        current_path = os.getcwd()
        current_path = os.path.dirname(current_path)
        data_path = os.path.join(current_path,"data",self.data,crypto,f"{crypto}-1m-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv")
        df = pl.read_csv(f"{data_path}")
        low = self.data_visitor_csv(df,"low",tidx)
        return np.nanmin(low)

class High(cache_maker):
    def __init__(self,interval,memmap_name,start,end,crypto,data):
        super().__init__(interval,memmap_name,start,end,crypto)
        self.data = data


    def calculate(self,tidx,crypto,year,month,day):
        current_path = os.getcwd()
        current_path = os.path.dirname(current_path)
        data_path = os.path.join(current_path,"data",self.data,crypto,f"{crypto}-1m-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv")
        df = pl.read_csv(f"{data_path}")
        high = self.data_visitor_csv(df,"high",tidx)
        print(np.nanmax(high))
        return np.nanmax(high)

class Return(cache_maker):
    def __init__(self,interval,memmap_name,start,end,crypto,data):
        super().__init__(interval,memmap_name,start,end,crypto)
        self.data = data


    def calculate(self,tidx,crypto,year,month,day):
        current_path = os.getcwd()
        current_path = os.path.dirname(current_path)
        data_path = os.path.join(current_path,"data",self.data,crypto,f"{crypto}-1m-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv")
        df = pl.read_csv(f"{data_path}")
        close = self.data_visitor_csv(df,"close",tidx)
        open = self.data_visitor_csv(df,"open",tidx)
        ret = close[-1]/open[0]-1
        print(ret)
        return ret

class Volume(cache_maker):
    def __init__(self,interval,memmap_name,start,end,crypto,data):
        super().__init__(interval,memmap_name,start,end,crypto)
        self.data = data

    def calculate(self,tidx,crypto,year,month,day):
        current_path = os.getcwd()
        current_path = os.path.dirname(current_path)
        data_path = os.path.join(current_path,"data",self.data,crypto,f"{crypto}-1m-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv")
        df = pl.read_csv(f"{data_path}")
        volume = self.data_visitor_csv(df,"volume",tidx)
        volume_mean = np.nanmean(volume)
        print(volume_mean)
        return volume_mean


class Close(cache_maker):
    def __init__(self,interval,memmap_name,start,end,crypto,data):
        super().__init__(interval,memmap_name,start,end,crypto)
        self.data = data


    def calculate(self,tidx,crypto,year,month,day):
        current_path = os.getcwd()
        current_path = os.path.dirname(current_path)
        data_path = os.path.join(current_path,"data",self.data,crypto,f"{crypto}-1m-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv")
        df = pl.read_csv(f"{data_path}")
        close = self.data_visitor_csv(df,"close",tidx)
        return close[-1]


class Open(cache_maker):
    def __init__(self,interval,memmap_name,start,end,crypto,data):
        super().__init__(interval,memmap_name,start,end,crypto)
        self.data = data


    def calculate(self,tidx,crypto,year,month,day):
        current_path = os.getcwd()
        current_path = os.path.dirname(current_path)
        data_path = os.path.join(current_path,"data",self.data,crypto,f"{crypto}-1m-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv")
        df = pl.read_csv(f"{data_path}")
        open = self.data_visitor_csv(df,"open",tidx)
        return open[0]

if __name__ == "__main__":
    start_date = datetime.date(2021, 11, 1)
    end_date = datetime.date(2023, 9, 1)
    cache = Volume(60,"Volume_1h",start_date,end_date,["ETHUSDT"],"kline")
    cache.whole_process()
    cache = Open(60,"Open_1h",start_date,end_date,["ETHUSDT"],"kline")
    cache.whole_process()
    cache = Close(60,"Close_1h",start_date,end_date,["ETHUSDT"],"kline")
    cache.whole_process()
    cache = Return(60,"Return_1h",start_date,end_date,["ETHUSDT"],"kline")
    cache.whole_process()
    cache = High(60,"High_1h",start_date,end_date,["ETHUSDT"],"kline")
    cache.whole_process()
    cache = Low(60,"Low_1h",start_date,end_date,["ETHUSDT"],"kline")
    cache.whole_process()
