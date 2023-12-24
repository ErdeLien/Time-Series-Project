import numpy as np

from utils import check_performance,html_performance
import pandas as pd


file = "BIAS_1h_daily.parquet"
file_name = "D:\\DDD\\Cache\\5_min\\long_short_5min"
shape = (670,1,288)
print(np.memmap(file_name,dtype = "float32",mode = 'r',shape = shape))
print(pd.read_parquet(f"D:\DDD\Pnl\{file}"))
check_performance(f"D:\DDD\Pnl\{file}")
# html_performance(f"D:\DDD\Pnl\{file}")