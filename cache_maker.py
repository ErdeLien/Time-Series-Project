
from abc import ABC,abstractmethod
from typing import List,Dict,Tuple
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import create_memmap,generate_dates_list
import polars as pl


#样本为左闭右开


class cache_maker(ABC):
    def __init__(self,interval:int,memmap_name,start,end:str,crypto:List[str]):
        self.interval = interval
        self.dates = generate_dates_list('2021-11-1', '2023-9-1')
        self.crypto = crypto
        if not os.path.exists(memmap_name):
            #此处不同间隔输入的shape不一样
            print(len(self.dates))
            create_memmap(memmap_name,np.float32,(len(self.dates),len(self.crypto),int(1440/self.interval)))
        self.memmap = np.memmap(memmap_name,'float32',mode='r+',shape=(len(self.dates),len(self.crypto),int(1440/self.interval)))
        self.start = self.dates.index(start)
        self.end = self.dates.index(end)

    @abstractmethod
    def calculate(self,didx,tidx,crypto,year):
        '''
        重写部分
        '''
        pass
        return

    def data_visitor_csv(self, data: pl.DataFrame, column: str, tidx: int) -> pl.DataFrame:
        '''
        从csv中按照一定间隔读取数据
        '''
        time_list = [i for i in range(0, 86340000 + 60001, 60000 * self.interval)]

        # 使用polars过滤数据
        filtered_data = data.filter(
            (data["open_time"] >= time_list[tidx]) & (data["open_time"] < time_list[tidx + 1])
        )

        return filtered_data.select(column).to_numpy()
    
    # def data_visitor_csv(self,data,column:str,tidx:int):
    #     '''
    #     从csv中按照一定间隔读取数据
    #     '''
    #     time_list = []
    #     for i in range(0,86340000+60001,60000*self.interval):
    #         time_list.append(i)
    #     return data[(time_list[tidx]<=data["Open time"]) & (data["Open time"]< time_list[tidx+1])][column]

    # def one_day_process(self, didx_tuple):
    #     didx = didx_tuple[0]
    #     didx_date = didx_tuple[1]
    #     year, month, day = didx_date.year, didx_date.month, didx_date.day
    #     for tidx in range(int(1440/self.interval)):
    #         for crypto_num,crypto in enumerate(self.crypto):
    #             cache_data = self.calculate(tidx,crypto,year,month,day)
    #             if not np.isnan(self.memmap[didx+self.start,crypto_num,tidx]):
    #                 pass
    #             else:
    #                 self.memmap[didx+self.start,crypto_num,tidx] = cache_data
    #     return

    #
    def one_day_process(self, didx_tuple):
        didx = didx_tuple[0]
        didx_date = didx_tuple[1]
        year, month, day = didx_date.year, didx_date.month, didx_date.day
        for tidx in range(int(1440 / self.interval)):
            for crypto_num, crypto in enumerate(self.crypto):
                try:
                    cache_data = self.calculate(tidx,crypto,year,month,day)
                    if not np.isnan(self.memmap[didx+self.start,crypto_num,tidx]):
                        pass
                    else:
                        self.memmap[didx+self.start,crypto_num,tidx] = cache_data
                except Exception as e:
                    print("Error")
        return
    
    def whole_process(self):
        for didx,didx_date in tqdm(enumerate(self.dates[self.start:self.end])):
            self.one_day_process((didx,didx_date))
        return


    # def whole_process(self,core=6):
    #     with ProcessPoolExecutor(max_workers = core) as executor:
    #         futures = [executor.submit(self.one_day_process,(didx,didx_date)) for didx,didx_date in enumerate(self.dates[self.start:self.end])]
    #         progress = tqdm(total=len(futures),desc="Processing")
    #         for futures in as_completed(futures):
    #             progress.update()
    #     progress.close()
    #     return


class close(cache_maker):
    def __init__(self,interval,memmap_name,start,end,data):
        super().__init__(interval,memmap_name,start,end)
        self.data = data

    def calculate(self,tidx,crypto,year,month,day):
        current_path = os.getcwd()
        data_path = os.path.join(current_path,"data",self.data,crypto,f"{crypto}-1m-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv")
        df = pl.read_csv(f"{data_path}")
        close = self.data_visitor_csv(df,"Close",tidx)
        close_end = close.iloc[-1]
        return close_end
    

class close_return(cache_maker):
    def __init__(self,interval,memmap_name,start,end,data,crypto):
        super().__init__(interval,memmap_name,start,end,crypto)
        self.data = data


    def calculate(self,tidx,crypto,year,month,day):
        current_path = os.getcwd()
        data_path = os.path.join(current_path,"data",self.data,crypto,f"{crypto}-1m-{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv")
        df = pl.read_csv(f"{data_path}")
        close = self.data_visitor_csv(df,"Close",tidx)
        open = self.data_visitor_csv(df,"Open",tidx)
        # quit()
        open_begin = open[0]
        close_end = close[-1]
        ret = (close_end-open_begin)/open_begin
        return ret
    


