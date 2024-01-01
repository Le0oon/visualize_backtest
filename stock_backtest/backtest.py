import sys
import pandas as pd
sys.path.append('/Users/liuqize/Documents/QuantAnalysis/QuantFrame')
import my_config as config
from Data import StockData

data_path = config.data_path

def load_data(start_dt, universe, rtype):
    
    DB = StockData(data_path,start_dt=-start_dt,update=False)
    df = DB.daily_df_qfq
    if universe == 'all':
        df = df.merge(
            DB.ret_df[['TradingDay','SecuCode',rtype]],
            on=['TradingDay','SecuCode'],how='left'
            )
    else:
        df = df.merge(
            DB.ret_df[['TradingDay','SecuCode',rtype,universe]],
            on=['TradingDay','SecuCode'],how='left'
            )
        df = df[df[universe].fillna(False)]

    
    return df

def calculate_input_expression(expression:str, df:pd.DataFrame):
    pass