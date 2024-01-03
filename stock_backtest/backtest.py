USE_ONLINE_DATA = False

import sys
import pandas as pd
import shutil
import os 
sys.path.append('/Users/liuqize/Documents/QuantAnalysis/QuantFrame')
import my_config as config
USE_ONLINE_DATA = True
if USE_ONLINE_DATA:
    from Data import StockData

from AutoAlpha.EvaluateAlpha import EvalExpressionTree
from StockBackTest import BackTest

data_path = config.data_path

def load_data(start_dt, end_dt,universe, rtype):
    
    DB = StockData(data_path,start_dt=start_dt,update=False)
    df = DB.daily_df_qfq
    ret_feature_cols = [
        'raw_close_close','raw_close_open','raw_close_high',
        'raw_close_low','raw_close_vwap','excess_return']
    if universe == 'all':
        df = df.merge(
            DB.ret_df[['TradingDay','SecuCode'] + ret_feature_cols],
            on=['TradingDay','SecuCode'],how='left'
            )
    else:
        df = df.merge(
            DB.ret_df[['TradingDay','SecuCode',universe] + ret_feature_cols],
            on=['TradingDay','SecuCode'],how='left'
            )
        df = df[df[universe].fillna(False)]
        
    ret_df = DB.ret_df[['TradingDay','SecuCode',rtype]].query(
        'TradingDay >= @start_dt & TradingDay <= @end_dt')
    df = df.query('TradingDay >= @start_dt & TradingDay <= @end_dt')
    
    return df, ret_df

def calculate_input_expression(expression:str, df:pd.DataFrame):
    et = EvalExpressionTree()
    et.initialize_with_expression(expression)
    factor_col = et.calc_tree(df)
    
    return df[['TradingDay','SecuCode',factor_col]].dropna(), factor_col

def main(start_dt,end_dt,universe,rtype,expression,turnover_fee=0.0005,factor_name=None):
    # print(start_dt,end_dt)
    save_dir = os.path.join('./static/output',factor_name)
    if os.path.exists(save_dir):
        # 如果已经存在因子，则删除之
        shutil.rmtree(save_dir)

    df,ret_df = load_data(start_dt,end_dt,universe,rtype)
    factor_df,factor_col = calculate_input_expression(expression,df)
    if factor_name is not None:
        factor_df.rename(columns={factor_col:factor_name},inplace=True)
        factor_col = factor_name
    
    BT = BackTest(factor_df,factor_col,save_dir='./static/output',plot=False,folder_name=factor_name)
    rst = BT.wrapped_analysis(ret_df,rtype=rtype,fee=turnover_fee,)
    
    return rst


if __name__ == "__main__":
    main('2020-01-01','2021-01-01',
         universe='all',
         rtype='excess_return',
         expression='comb_div(open,close)',
         turnover_fee=0.0005,
         factor_name='test_factor'
         )