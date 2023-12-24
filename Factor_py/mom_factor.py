import sys
import os
sys.path.append(os.path.abspath("./"))
import ts_calculator as tc
from utils import data_visitor,data_visitor_backtest
import numpy as np
import datetime
from tqdm import tqdm
from config import FACTOR_CONFIG
sys.path.append(os.path.abspath("../Factor_maker"))
from Factor_subclass import Factor_continuous,Factor_discrete
from utils import check_performance,html_performance



class Corr(Factor_continuous):
    def __init__(self,asset_ret_path,crypto,alpha,start,end,shape,pnl_path,back_day,fee,high_thre,low_thre,para_3):

        super().__init__(asset_ret_path,crypto,alpha,start,end,shape,pnl_path,back_day,fee,high_thre,low_thre)
        self.path = os.getcwd()
        self.para_1 = os.path.join(os.path.dirname(self.path),"Cache\\1_h\\Volume_1h")
        self.para_2 = os.path.join(os.path.dirname(self.path),"Cache\\1_h\\Close_1h")
        self.para_3 = para_3


    def calculate(self, didx, tidx):
        volume = data_visitor(self.para_1,didx,tidx,self.para_3,0)
        close = data_visitor(self.para_2,didx,tidx,self.para_3,0)
        corr = tc.correlation(tc.ts_zscore(volume),tc.ts_zscore(close))
        return corr


class BIAS(Factor_continuous):
    def __init__(self,asset_ret_path,crypto,alpha,start,end,shape,pnl_path,back_day,fee,high_thre,low_thre,para_2,para_3):

        super().__init__(asset_ret_path,crypto,alpha,start,end,shape,pnl_path,back_day,fee,high_thre,low_thre)
        self.path = os.getcwd()
        self.para_1 = os.path.join(os.path.dirname(self.path),"Cache\\1_h\\Close_1h")
        self.para_2 = para_2
        self.para_3 = para_3

    def calculate(self, didx, tidx):
        close_short = data_visitor(self.para_1,didx,tidx,self.para_2,0)
        close_long = data_visitor(self.para_1,didx,tidx,self.para_3,0)
        close_short_ewa = tc.nioewa(close_short)
        close_long_ewa = tc.nioewa(close_long)
        bias = -close_short_ewa+close_long_ewa
        return bias

class Long_Short(Factor_continuous):
    def __init__(self,asset_ret_path,crypto,alpha,start,end,shape,pnl_path,back_day,fee,high_thre,low_thre,para_2,para_3):

        super().__init__(asset_ret_path,crypto,alpha,start,end,shape,pnl_path,back_day,fee,high_thre,low_thre)
        self.path = os.getcwd()
        self.para_1 = os.path.join(os.path.dirname(self.path),"Cache\\5_min\\long_short_5min")
        self.para_2 = para_2
        self.para_3 = para_3

    def calculate(self, didx, tidx):
        long_short = data_visitor(self.para_1,didx,tidx,self.para_2,self.para_3,(670,1,288))
        ema_long_short = tc.nioewa(long_short)
        return -ema_long_short


if __name__ == "__main__":
    pnl_path = FACTOR_CONFIG["pnl_path"]
    alpha_path = FACTOR_CONFIG["alpha_path"]
    cache_path = FACTOR_CONFIG["cache_path"]
    start_date = FACTOR_CONFIG["start_date"]
    end_date = FACTOR_CONFIG["end_date"]
    back_days = FACTOR_CONFIG["back_days"]
    fee = FACTOR_CONFIG["fee"]
    high_thre = FACTOR_CONFIG["high_thre"]
    low_thre = FACTOR_CONFIG["low_thre"]
    alpha_name = FACTOR_CONFIG["alpha_name"]
    asset_ret_path = FACTOR_CONFIG["asset_ret_path"]
    crypto = FACTOR_CONFIG["crypto"]
    shape = FACTOR_CONFIG["shape"]
    para_3 = 5
    para_2 = 35
    # for para_2 in range(50,100):
    #     alpha_name = f"BIAS_{para_2}_1h"
    #     factor = BIAS(asset_ret_path=asset_ret_path,crypto=crypto,alpha=alpha_name,start=start_date,end = end_date,shape = shape,pnl_path = pnl_path,
    #               back_day=back_days,fee = fee,high_thre=high_thre,low_thre=low_thre,para_2 = para_2, para_3 = para_3)
    #     file = f"BIAS_{para_2}_1h_daily.parquet"
    #     factor.whole_process()
    #     print(f"Para_2 is {para_2}")
    #     check_performance(f"D:\DDD\Pnl\{alpha_name}_daily.parquet")
    # alpha_name = f"Long_Short_Ratio"
    # factor = Long_Short(asset_ret_path=asset_ret_path,crypto=crypto,alpha=alpha_name,start=start_date,end = end_date,shape = shape,pnl_path = pnl_path,
    #               back_day=back_days,fee = fee,high_thre=high_thre,low_thre=low_thre,para_2 = 35, para_3 = 5)
    # factor.whole_process()
    # factor.store_alpha(alpha_path)
    # factor.store_alpha_zscore(alpha_path)
    # factor.store_position(alpha_path)

    # alpha_name = f"BIAS"
    # factor = BIAS(asset_ret_path=asset_ret_path,crypto=crypto,alpha=alpha_name,start=start_date,end = end_date,shape = shape,pnl_path = pnl_path,
    #               back_day=back_days,fee = fee,high_thre=high_thre,low_thre=low_thre,para_2 = 228, para_3 = 500)
    # factor.whole_process()
    # factor.store_alpha(alpha_path)
    # factor.store_alpha_zscore(alpha_path)
    # factor.store_position(alpha_path)

    alpha_name = f"PV_Corr"
    factor = Corr(asset_ret_path=asset_ret_path,crypto=crypto,alpha=alpha_name,start=start_date,end = end_date,shape = shape,pnl_path = pnl_path,
                  back_day=back_days,fee = fee,high_thre=high_thre,low_thre=low_thre, para_3 = 194)
    factor.whole_process()
    factor.store_alpha(alpha_path)
    factor.store_alpha_zscore(alpha_path)
    factor.store_position(alpha_path)
