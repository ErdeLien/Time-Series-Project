import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import reshape_to_2d,shift_up
from utils import generate_dates_list,transfer_date_index
import datetime
'''
画图：
pnl,带fee pnl,drawdown,K线及买卖点，fee
画表：
分年份年化，波动率，夏普，tvr，带fee return,margin(每笔平均盈利)，交易次数，胜率，maxdrawdown
'''

class Factor_plot:
    def __init__(self,position,asset_return,periods_per_year=365*96,fee=0.0011):
        self.position = position
        self.asset_return = asset_return
        self.periods_per_year = periods_per_year
        self.fee = fee


    def pnl_plot(self):
        portfolio_returns = self.position * self.asset_return
        total_return = (1 + portfolio_returns).cumprod()
        transaction_cost = self.fee * abs(np.diff(self.position,axis=0))
        portfolio_returns_after_costs = portfolio_returns[:-1] - transaction_cost
        # 计算考虑费用后的累积PnL
        total_return_after_costs = (1 + portfolio_returns_after_costs).cumprod()
        # 绘制图形
        plt.plot(total_return, label='PnL without Costs')
        plt.plot(total_return_after_costs, label='PnL with Costs')
        plt.title("Cumulative PnL")
        plt.grid(True)
        plt.legend()
        plt.show()



    # def klin_trade_plot(self):

    # def table_plot(self):

    # def plot_all(self):
    
    # def save_plot(self):

if __name__ == "__main__":
    return_array = np.memmap("./Cache/Return",dtype="float32",mode="r",shape=(2800,1,96))
    position_array = np.memmap(f"./Factor/Mom_decay_10_5_position",dtype="float32",mode="r",shape=(2800,1,96))
    start_date = datetime.date(2018, 1, 1)
    end_date = datetime.date(2023, 8, 31)
    start_index = transfer_date_index(start_date)
    end_index = transfer_date_index(end_date)
    timestampe = generate_dates_list('2018-1-1', '2023-8-31')
    timestampe = np.repeat(timestampe, 96)
    return_2d = reshape_to_2d(return_array[start_index:end_index])
    return_lag = shift_up(return_2d)
    position_2d = reshape_to_2d(position_array[start_index:end_index])
    plot = Factor_plot(position_2d,return_lag)
    plot.pnl_plot()