import numpy as np
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
from ts_calculator import niomean,niostddev, threshold
from Factor import Factor
import sys
sys.path.append(os.path.abspath("./"))
from utils import array_shift,generate_dates_list,data_visitor,data_visitor_backtest
from typing import List


class Factor_continuous(Factor):
    def __init__(self,asset_ret_path,crypto:List[str],alpha,start,end,shape,pnl_path,back_day,fee,high_thre,low_thre):
        super().__init__(asset_ret_path,crypto,alpha,start,end,shape,pnl_path,fee)
        self.alpha_array = np.full((self.shape[0],self.shape[1],self.shape[2]),np.nan)
        self.alpha_zscore = np.full((self.shape[0],self.shape[1],self.shape[2]),np.nan)
        self.shape = shape
        self.back_day = back_day*shape[2]
        self.high_thre = high_thre
        self.low_thre = low_thre

    def pnl_calculate(self,didx,tidx):
        '''
        calculate pnl and store in a parquet file
        '''
        ret_array = np.memmap(self.asset_ret_path,'float32',mode='r+',shape=self.shape)
        alpha = self.calculate(didx,tidx)
        hist_alpha = data_visitor_backtest(self.alpha_array,didx,tidx,self.back_day+1,1)
        if didx < self.back_day//self.shape[2]:
            alpha_zscore = np.nan
            position = np.nan
        else:
            alpha_zscore = (alpha-niomean(hist_alpha))/niostddev(hist_alpha)
            position = threshold(alpha_zscore,self.high_thre,self.low_thre)
        last_position = data_visitor_backtest(self.position_array,didx,tidx,1,1)
        factor_ret = (position*array_shift(ret_array,didx,tidx,1))[0]-np.abs(position-last_position)*self.fee
        factor_ret_without_fee = (position*array_shift(ret_array,didx,tidx,1))[0]
        return factor_ret,alpha_zscore,position,alpha,factor_ret_without_fee


    def one_day_process(self,didx):
        for tidx in range(self.shape[2]):
            pnl,alpha_zscore,position,alpha,pnl_without_fee = self.pnl_calculate(didx+self.start,tidx)
            self.alpha_array[didx+self.start,:,tidx] = alpha
            if np.isnan(alpha_zscore):
                if tidx == 0:
                    self.alpha_zscore[didx+self.start,:,tidx] = self.alpha_zscore[didx+self.start-1,:,self.shape[2]-1]
                else:
                    self.alpha_zscore[didx + self.start, :, tidx] = self.alpha_zscore[didx + self.start - 1, :, tidx-1]
            else:
                self.alpha_zscore[didx+self.start,:,tidx] = alpha_zscore
            self.position_array[didx+self.start,:,tidx] = position
            self.pnl[didx+self.start,:,tidx] = pnl
            self.defee_pnl[didx+self.start,:,tidx] = pnl_without_fee
        return
    

    def store_alpha(self,alpha_path):
        alpha_array = np.memmap(f"{alpha_path}/{self.alpha}",dtype="float32",mode="w+",shape=self.shape)
        alpha_array[:] = self.alpha_array
        return
    

    def store_alpha_zscore(self,alpha_path):
        zscore_array = np.memmap(f"{alpha_path}/{self.alpha}_zscore",dtype="float32",mode="w+",shape=self.shape)
        zscore_array[:] = self.alpha_zscore
        return


class Factor_discrete(Factor):
    def __init__(self,asset_ret_path,crypto:List[str],alpha,start,end,shape,pnl_path,fee):
        super().__init__(asset_ret_path,crypto,alpha,start,end,shape,pnl_path,fee)
    

    def pnl_calculate(self,didx,tidx):
        ret_array = np.memmap(self.asset_ret_path,'float32',mode='r+',shape=self.shape)
        position = self.calculate(didx,tidx)
        last_position = data_visitor_backtest(self.position_array,didx,tidx,1,1)
        factor_ret = (position*array_shift(ret_array,didx,tidx,1))[0]-np.abs(position-last_position)*self.fee
        factor_ret_without_fee = (position*array_shift(ret_array,didx,tidx,1))[0]
        return factor_ret,position,factor_ret_without_fee
    
    def one_day_process(self,didx):
        for tidx in range(self.shape[2]):
            pnl,position,pnl_without_fee = self.pnl_calculate(didx+self.start,tidx)
            self.position_array[didx+self.start,:tidx] = position
            self.pnl[didx+self.start,:,tidx] = pnl
            self.defee_pnl[didx+self.start,:,tidx] = pnl_without_fee
        return