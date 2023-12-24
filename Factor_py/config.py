import os
import datetime
FACTOR_CONFIG = {
    #改因子时需要修改
    "alpha_name": "BIAS_1h",
    "start_date":datetime.date(2021,11,1),
    "end_date":datetime.date(2023,9,1),
    "shape":(670,1,24),
    "pnl_path":os.path.join(os.path.dirname(os.getcwd()),"Time-Series-Project","Pnl"),
    "alpha_path":os.path.join(os.path.dirname(os.getcwd()),"Time-Series-Project","Factor"),
    "cache_path":os.path.join(os.path.dirname(os.getcwd()),"Time-Series-Project","1_h\\Cache"),
    "back_days":30,
    "fee":0.0005,
    "high_thre":1,
    "low_thre":-1,
    #"asset_ret_path":os.path.join(os.path.dirname(os.getcwd()),"Return_1h"),
    "asset_ret_path":os.path.join(os.path.dirname(os.getcwd()),"Time-Series-Project","Return_1h"),
    #换asset时需要修改
    "crypto":["ETHUSDT"]
}
