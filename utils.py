from datetime import datetime, timedelta
import numpy as np
import pickle
import datetime as dt

import pandas as pd
import quantstats as qs

def check_performance(return_file):
    '''
    生成因子的sharpe ratio,return 和maximum drawdown
    '''
    df = pd.read_parquet(return_file)
    df.set_index("Date",inplace=True)
    df.index = pd.to_datetime(df.index)
    sharpe_Pnl = qs.stats.sharpe(df["Pnl"],periods = 365)
    sharpe_defee_Pnl = qs.stats.sharpe(df["Pnl_defee"],periods = 365)
    max_drawdown_Pnl = qs.stats.max_drawdown((1+df["Pnl"]).cumprod())
    max_drawdown_defee_Pnl = qs.stats.max_drawdown((1+df["Pnl_defee"]).cumprod())
    print(f"Pnl with fee max_drawdown is: {max_drawdown_Pnl}")
    print(f"Pnl without fee max_drawdown is: {max_drawdown_defee_Pnl}")
    print(f"Pnl with fee Sharpe is: {sharpe_Pnl}")
    print(f"Pnl without fee Sharpe is: {sharpe_defee_Pnl}")
    return


def html_performance(return_file):
    df = pd.read_parquet(return_file)
    df.set_index("Date",inplace=True)
    df.index = pd.to_datetime(df.index)
    # qs.reports.html(df["Pnl"],"factor_return",periods_per_year=365,)
    qs.plots.snapshot(df["Pnl"], title='Facebook Performance', show=True)
    # # qs.reports.html(df["Pnl_defee"],"factor_return",periods_per_year=365)
    # return


def check_memmap(memmap_file,shape):
    '''
    输出memmap中的数据
    '''
    np_array = np.memmap(memmap_file,'float32',mode='r',shape=shape)
    print(np_array)
    return

def data_visitor_backtest(data,didx,tidx,nIntv,niBefore,shape=(2222,1,24)):
    '''
    用来取cache中的数据，shape是需要提取的cache的shape
    '''
    interval_num = shape[2]
    nIntv_ap = nIntv-1
    whole_time = nIntv_ap + niBefore
    tidx_diff = tidx - whole_time
    tidx_diff_before = tidx - niBefore
    if tidx_diff>= 0 and tidx_diff_before >=0:
        didx_start = didx
        didx_end = didx
        tidx_start = tidx_diff
        tidx_end = tidx_diff_before + 1
    elif tidx_diff <0 and tidx_diff_before >= 0:
        didx_start = int(didx + ((tidx_diff-0.01)//interval_num))
        tidx_start = interval_num-(-tidx_diff%interval_num)
        didx_end = didx
        tidx_end = tidx_diff_before + 1
    else:
        didx_start = int(didx+((tidx_diff-0.01)//interval_num))
        tidx_start = interval_num-(-tidx_diff%interval_num)
        didx_end = didx + (tidx_diff_before//interval_num)
        tidx_end = interval_num-(-tidx_diff_before%interval_num)+1

    if didx_start == didx_end:
        out_data = data[didx_end,:,tidx_start:tidx_end]
    elif didx_end - didx_start == 1:
        before_data = data[didx_start,:,tidx_start:]
        after_data = data[didx_end,:,:tidx_end]
        out_data = np.hstack((before_data,after_data))
    else:
        before_data = data[didx_start,:,tidx_start:]
        mid_data = data[didx_start+1:didx_end,:,:]
        mid_data = data[didx_start+1:didx_end,:,:]
        mid_data_2d = mid_data.reshape(mid_data.shape[1],-1)
        after_data = data[didx_end,:,:tidx_end]
        out_data = np.hstack((before_data,mid_data_2d,after_data))
    return out_data.T



def data_visitor(filename,didx,tidx,nIntv,niBefore,shape=(670,1,24)):
    '''
    用来取cache中的数据，shape是需要提取的cache的shape
    '''
    data = np.memmap(filename,dtype='float32',mode='r',shape=shape)
    interval_num = shape[2]
    nIntv_ap = nIntv-1
    whole_time = nIntv_ap + niBefore
    tidx_diff = tidx - whole_time
    tidx_diff_before = tidx - niBefore
    if tidx_diff>= 0 and tidx_diff_before >=0:
        didx_start = didx
        didx_end = didx
        tidx_start = tidx_diff
        tidx_end = tidx_diff_before + 1
    elif tidx_diff <0 and tidx_diff_before >= 0:
        didx_start = int(didx + ((tidx_diff-0.01)//interval_num))
        tidx_start = interval_num-(-tidx_diff%interval_num)
        didx_end = didx
        tidx_end = tidx_diff_before + 1
    else:
        didx_start = int(didx+((tidx_diff-0.01)//interval_num))
        tidx_start = interval_num-(-tidx_diff%interval_num)
        didx_end = didx + (tidx_diff_before//interval_num)
        tidx_end = interval_num-(-tidx_diff_before%interval_num)+1
    if didx_start == didx_end:
        out_data = data[didx_end,:,tidx_start:tidx_end]
    elif didx_end - didx_start == 1:
        before_data = data[didx_start,:,tidx_start:]
        after_data = data[didx_end,:,:tidx_end]
        out_data = np.hstack((before_data,after_data))
    else:
        before_data = data[didx_start,:,tidx_start:]
        mid_data = data[didx_start+1:didx_end,:,:]
        mid_data = data[didx_start+1:didx_end,:,:]
        mid_data_2d = mid_data.reshape(mid_data.shape[1],-1)
        after_data = data[didx_end,:,:tidx_end]
        out_data = np.hstack((before_data,mid_data_2d,after_data))
    return out_data.T


def create_memmap(filename,dtype,shape):
    mmaped_data = np.memmap(filename,dtype=dtype,mode='w+',shape=shape)
    mmaped_data[:] = np.nan
    return


def generate_dates_list(start_date, end_date):
    """
    Generate a list of dates between start_date and end_date (inclusive).

    Parameters:
    - start_date (str): Start date in the format 'YYYY-MM-DD'
    - end_date (str): End date in the format 'YYYY-MM-DD'

    Returns:
    - dates (list): List of dates between start_date and end_date, including both
    """
    # Generate date range using pandas
    dates = pd.date_range(start=start_date, end=end_date).tolist()

    # Convert Timestamps to date objects
    dates_list = [date.date() for date in dates]

    return dates_list


def transfer_date_index(datetime):
    '''
    输入日期返回memmap对应的didx
    '''
    dates = generate_dates_list('2016-1-1', '2023-9-1')
    index = dates.index(datetime)
    return index


def save_model(model, model_name):
    filename = f"params/{model_name}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def load_model(model_name):
    filename = f"params/{model_name}.pkl"
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def shift_up(array, fill_value=np.nan):
    """将2D numpy数组整体向上移动一格"""
    shifted_array = np.roll(array, shift=-1, axis=0)
    shifted_array[-1, :] = fill_value
    return shifted_array


def array_shift(arr, didx, tidx, shift):
    # 验证didx和tidx是否在正确的范围内
    if didx < 0 or didx >= arr.shape[0]:
        raise ValueError(f"didx ({didx}) out of bounds for dimension 0 with size {arr.shape[0]}")
    if tidx < 0 or tidx >= arr.shape[2]:
        raise ValueError(f"tidx ({tidx}) out of bounds for dimension 2 with size {arr.shape[2]}")

    new_tidx = tidx + shift
    new_didx = didx
    
    if new_tidx >= arr.shape[2]:
        # Compute how many times we've exceeded the tidx dimension
        overflow = new_tidx // arr.shape[2]
        
        # Adjust the didx by the overflow
        new_didx = (didx + overflow) % arr.shape[0]
        
        # Reset tidx to the remainder
        new_tidx = new_tidx % arr.shape[2]
    
    return arr[new_didx, :, new_tidx]


def reshape_to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError("Expected a 3D array")
    
    # 获取数组的形状 
    trading_days, assets, trading_periods = arr.shape
    
    # 重塑数组
    reshaped_arr = arr.transpose(0, 2, 1).reshape(trading_days*trading_periods, assets)
    
    return reshaped_arr


def generate_time_list(start_date, end_date, interval=15, interval_unit='minutes'):
    # Convert start_date and end_date to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    time_list = []

    # Calculate interval as a timedelta
    if interval_unit == "days":
        delta = timedelta(days=interval)
    elif interval_unit == "hours":
        delta = timedelta(hours=interval)
    elif interval_unit == "minutes":
        delta = timedelta(minutes=interval)
    elif interval_unit == "seconds":
        delta = timedelta(seconds=interval)
    else:
        return "Invalid interval unit"

    current_time = start_date

    while current_time <= end_date:
        # Append to list in your desired string format
        time_list.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))

        # Move to the next time point
        current_time += delta

    return time_list


# Test the function
"""start_date = "2023-09-22"
end_date = "2023-09-22"
interval = 10
interval_unit = "seconds"

time_list = generate_time_list(start_date, end_date, interval, interval_unit)
print(time_list)"""
