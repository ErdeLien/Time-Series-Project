#分三步处理，第一步：计算出原始alpha
#第二步：计算出处理后的alpha并且做归一化处理和映射处理
#第三步：根据映射处理过后的因子计算仓位和return还有pnl

import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
from ts_calculator import niomean,niostddev, threshold
import sys
sys.path.append(os.path.abspath("./"))
from utils import array_shift,generate_dates_list,data_visitor,data_visitor_backtest
from typing import List
# def generate_dates_list(start_date, end_date):
#     """
#     Generate a list of dates between start_date and end_date (inclusive).
    
#     Parameters:
#     - start_date (str): Start date in the format 'YYYY-MM-DD'
#     - end_date (str): End date in the format 'YYYY-MM-DD'
    
#     Returns:
#     - dates (list): List of dates between start_date and end_date
#     """
#     start = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
#     end = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
#     delta = datetime.timedelta(days=1)
#     current_date = start
#     dates_list = []
#     while current_date < end:
#         dates_list.append(current_date)
#         current_date += delta

#     return dates_list


class Factor(ABC):
    """
    Base Factor class that represents a generic factor.
    """

    def __init__(self,asset_ret_path,crypto:List[str],alpha,start,end,shape,pnl_path,fee):
        """
        Initializes the Factor with given data.

        Args:
        - data (pd.DataFrame): A dataframe with time as rows and assets as columns.
        """

        # if not isinstance(data, pd.DataFrame):
        #     raise ValueError("Data should be a pandas DataFrame.")
        self.asset_ret_path = asset_ret_path
        self.pnl_path = pnl_path
        self.alpha = alpha
        self.dates = generate_dates_list('2021-11-1', '2023-9-1')
        self.shape = shape
        self.crypto = crypto
        self.nInstruments = len(self.crypto)
        self.pnl = np.full((self.shape[0],self.shape[1],self.shape[2]),np.nan)
        self.defee_pnl = np.full((self.shape[0],self.shape[1],self.shape[2]),np.nan)
        # self.alpha_array = np.full((self.shape[0],self.shape[1],self.shape[2]),np.nan)
        # self.alpha_zscore = np.full((self.shape[0],self.shape[1],self.shape[2]),np.nan)
        self.position_array = np.full((self.shape[0],self.shape[1],self.shape[2]),np.nan)
        self.start = self.dates.index(start)
        self.end = self.dates.index(end)
        self.fee = fee
        #示例
        #self.para_1 = "Return"
        #self.para_2 = ...
        #self.para_3 = ...
        #之后在calculate中调用


    @abstractmethod
    def calculate(self,didx,tidx):
        """
        Calculates the factor value.
        Must be overridden by subclasses.
        """
        pass
        return


    @abstractmethod
    def pnl_calculate(self,didx,tidx):
        '''
        calculate pnl and store in a parquet file
        '''
        pass
        return

    # def pnl_calculate(self,didx,tidx):
    #     '''
    #     calculate pnl and store in a parquet file
    #     '''
    #     ret_array = np.memmap(self.asset_ret_path,'float32',mode='r+',shape=self.shape)
    #     alpha = self.calculate(didx,tidx)
    #     hist_alpha = data_visitor_backtest(self.alpha_array,didx,tidx,self.back_day+1,1)
    #     alpha_zscore = (alpha-niomean(hist_alpha))/niostddev(hist_alpha)
    #     position = threshold(alpha_zscore,self.high_thre,self.low_thre)
    #     last_position = data_visitor_backtest(self.position_array,didx,tidx,1,1)
    #     factor_ret = (position*array_shift(ret_array,didx,tidx,1))[0]-np.abs(position-last_position)*self.fee
    #     factor_ret_without_fee = (position*array_shift(ret_array,didx,tidx,1))[0]
    #     return factor_ret,alpha_zscore,position,alpha,factor_ret_without_fee


    # def pnl_calculate_position(self,didx,tidx):
    #     ret_array = np.memmap(self.asset_ret_path,'float32',mode='r+',shape=self.shape)
    #     position = self.calculate(didx,tidx)
    #     last_position = data_visitor_backtest(self.position_array,didx,tidx,1,1)
    #     factor_ret = (position*array_shift(ret_array,didx,tidx,1))[0]-np.abs(position-last_position)*self.fee
    #     factor_ret_without_fee = (position*array_shift(ret_array,didx,tidx,1))[0]
    #     return factor_ret,position,factor_ret_without_fee
    
    @abstractmethod
    def one_day_process(self,didx):
        pass
        return


    # def one_day_process(self,didx):
    #     for tidx in range(self.shape[2]):
    #         pnl,alpha_zscore,position,alpha,pnl_without_fee = self.pnl_calculate(didx+self.start,tidx)
    #         self.alpha_array[didx+self.start,:,tidx] = alpha
    #         self.alpha_zscore[didx+self.start,:,tidx] = alpha_zscore
    #         self.position_array[didx+self.start,:,tidx] = position
    #         self.pnl[didx+self.start,:,tidx] = pnl
    #         self.defee_pnl[didx+self.start,:,tidx] = pnl_without_fee
    #     return

    # def one_day_process(self,didx):
    #     for tidx in range(self.shape[2]):
    #         try:
    #             pnl,alpha_zscore,position,alpha = self.pnl_calculate(didx+self.start,tidx)
    #             print(pnl)
    #             self.alpha_array[didx+self.start,:,tidx] = alpha
    #             self.alpha_zscore[didx+self.start,:,tidx] = alpha_zscore
    #             self.position[didx+self.start,:,tidx] = position
    #             self.pnl[didx+self.start,:,tidx] = pnl
    #         except Exception as e:
    #             print("Error") 
    #     return


    def store_interval_parquet(self,interval):
        pnl = self.pnl[:,:,interval]
        pnl_defee = self.defee_pnl[:,:,interval]
        dates = np.array(self.dates)
        pnl_array = np.column_stack((dates,pnl,pnl_defee))
        pnl_df = pd.DataFrame(pnl_array,columns=["Date","Pnl","Pnl_defee"])
        path = os.path.join(self.pnl_path,f"{self.alpha}_{interval}.parquet")
        pnl_df.to_parquet(path,engine="pyarrow")
        return
    

    def store_whole_parquet(self):
        whole_pnl = np.empty((self.shape[0],self.shape[1]))
        whole_pnl_defee = np.empty((self.shape[0],self.shape[1]))
        for didx in range(self.shape[0]):
            for crypto_num, _ in enumerate(self.crypto):
                returns = np.nan_to_num(self.pnl[didx,crypto_num,:])
                returns_defee = np.nan_to_num(self.defee_pnl[didx,crypto_num,:])
                daily_pnl = np.cumprod(returns+1) - 1
                daily_pnl_defee = np.cumprod(returns_defee+1) - 1
                whole_pnl[didx,crypto_num] = daily_pnl[-1]
                whole_pnl_defee[didx,crypto_num] = daily_pnl_defee[-1]
        dates = np.array(self.dates)
        pnl_array = np.column_stack((dates,whole_pnl,whole_pnl_defee))
        pnl_df = pd.DataFrame(pnl_array,columns=["Date","Pnl","Pnl_defee"])
        path = os.path.join(self.pnl_path,f"{self.alpha}_daily.parquet")
        pnl_df.to_parquet(path,engine="pyarrow")
        return
    

    def store_parquet(self):
        for tidx in range(self.shape[2]):
            self.store_interval_parquet(tidx)
        self.store_whole_parquet()
        return
    

    # def store_alpha(self,alpha_path):
    #     alpha_array = np.memmap(f"{alpha_path}/{self.alpha}",dtype="float32",mode="w+",shape=self.shape)
    #     alpha_array[:] = self.alpha_array
    #     return
    

    # def store_alpha_zscore(self,alpha_path):
    #     zscore_array = np.memmap(f"{alpha_path}/{self.alpha}_zscore",dtype="float32",mode="w+",shape=self.shape)
    #     zscore_array[:] = self.alpha_zscore
    #     return


    def store_position(self,alpha_path):
        position_array = np.memmap(f"{alpha_path}/{self.alpha}_position",dtype="float32",mode="w+",shape=self.shape)
        position_array[:] = self.position_array
        return

    def whole_process(self,core=6):
        for didx,didx_date in tqdm(enumerate(self.dates[self.start:self.end])):
            self.one_day_process(didx)
        self.store_parquet()
        return
    
    # def whole_process(self,core=6):
    #     with ProcessPoolExecutor(max_workers = core) as executor:
    #         futures = [executor.submit(self.one_day_process,didx) for didx,didx_date in enumerate(self.dates[self.start:self.end])]
    #         progress = tqdm(total=len(futures),desc="Processing")
    #         for futures in as_completed(futures):
    #             progress.update()
    #     progress.close()
    #     self.store_parquet()
    #     return

