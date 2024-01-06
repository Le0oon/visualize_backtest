# Author: Qize Liu
# Description: Document of Operator Information
from AutoAlpha import SingleAttrTsOperator, SingleAttrOperator, DoubleAttrOperator
import pandas as pd
from AutoAlpha.column_type_check import *


SUB_POP_SIZE = 1000

"""
types:
  - price: 价格数据
  - ticker_float: ticker 自身数据，比如volume, turnover, 什么数据都可以往里装
  - pct_change: 变化率数据
  - uniform: [-1,1] 或 [0,1]
  - norm: 正态数据, uniform 变量进行加减乘除后视为norm 变量
  - float: 时序、截面均可比的浮点数
"""
# 数据列信息, index: column name, columns: weight, type
column_info = pd.DataFrame({
    'open': {'weight': 1, 'type': 'price'},
    'high': {'weight': 1, 'type': 'price'},
    'low': {'weight': 1, 'type': 'price'},
    'close': {'weight': 1, 'type': 'price'},
    'TurnoverVolume': {'weight': 1, 'type': 'ticker_float'},
    'TurnoverValue': {'weight': 1, 'type': 'ticker_float'},
    'vwap': {'weight': 1, 'type': 'price'},
    'raw_close_close': {'weight': 1, 'type': 'pct_change'},
    'raw_close_open': {'weight': 1, 'type': 'pct_change'},
    'raw_close_high': {'weight': 1, 'type': 'pct_change'},
    'raw_close_low': {'weight': 1, 'type': 'pct_change'},
    'raw_close_vwap': {'weight': 1, 'type': 'pct_change'},
}).T
column_info['weight'] = column_info['weight'].astype(float)

integer_info = pd.DataFrame({
    # 1,2,...,19
    **{i: {'weight': 1, 'type': 'int'} for i in range(1, 20)},
    # 20, 10, 15,..., 45
    **{i*5: {'weight': 1, 'type': 'int'} for i in range(4, 10)},
    # 50,60,..., 100
    **{i*10: {'weight': 1, 'type': 'int'} for i in range(5, 11)},
}).T

integer_info['weight'] = integer_info['weight'].astype(float)
integer_info.index = integer_info.index.astype(int)

# func name to func mapping
funcName2func = {
    **{v.__func__.__name__: v.__func__ for _, v in vars(
        SingleAttrTsOperator).items() if isinstance(v, staticmethod)},
    **{v.__func__.__name__: v.__func__ for _, v in vars(
        SingleAttrOperator).items() if isinstance(v, staticmethod)},
    **{v.__func__.__name__: v.__func__ for _, v in vars(
        DoubleAttrOperator).items() if isinstance(v, staticmethod)}

}

# func name to func mapping
funcName2className = {
    **{v.__func__.__name__: 'SingleAttrTsOperator' for _, v in vars(
        SingleAttrTsOperator).items() if isinstance(v, staticmethod)},
    **{v.__func__.__name__: 'SingleAttrOperator' for _, v in vars(
        SingleAttrOperator).items() if isinstance(v, staticmethod)},
    **{v.__func__.__name__: 'DoubleAttrOperator' for _, v in vars(
        DoubleAttrOperator).items() if isinstance(v, staticmethod)}

}

# func name list
funcs_df = pd.DataFrame({
    # 单变量时序操作符
    'ts_mean': {'weight': 5, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_std': {'weight': 5, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_max': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_min': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_median': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_skew': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('float')},
    'ts_kurt': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('float')},
    'ts_autocorr': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('uniform')},
    'ts_shift': {'weight': 10, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_diff': {'weight': 10, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_median_abs_deviation': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_rms': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': IDENTICAL_MAP()},
    'ts_norm_mean': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('norm')},
    'ts_norm_min': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('norm')},
    'ts_norm_max': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('norm')},
    'ts_norm_min_max': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('norm')},
    'ts_ratio_beyond_3sigma': {'weight': 0.5, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('uniform')},
    'ts_ratio_beyond_2sigma': {'weight': 0.5, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('uniform')},
    'ts_index_mass_median': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('uniform')},
    'ts_number_cross_mean': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('float')},
    'ts_time_asymmetry_stats': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('float')},
    'ts_longest_strike_above_mean': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('float')},
    'ts_longest_strike_below_mean': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('float')},
    'ts_mean_over_1norm': {'weight': 1, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('norm')},
    'ts_norm': {'weight': 3, 'type': 'SingleAttrTsOperator', 'value_map': CONSTANT_MAP('norm')},

    # 单变量操作符
    'cs_rank': {'weight': 2, 'type': 'SingleAttrOperator', 'value_map': CONSTANT_MAP('uniform')},
    'ind_neu': {'weight': 0.5, 'type': 'SingleAttrOperator', 'value_map': IDENTICAL_MAP()},
    'cs_norm': {'weight': 2, 'type': 'SingleAttrOperator', 'value_map': CONSTANT_MAP('norm')},

    # 双变量操作符
    'comb_add': {'weight': 10, 'type': 'DoubleAttrOperator', 'value_map': ADD_SUB_MAP()},
    'comb_sub': {'weight': 10, 'type': 'DoubleAttrOperator', 'value_map': ADD_SUB_MAP()},
    'comb_mul': {'weight': 10, 'type': 'DoubleAttrOperator', 'value_map': MUL_MAP()},
    'comb_div': {'weight': 10, 'type': 'DoubleAttrOperator', 'value_map': DIV_MAP()},
    'ts_corr_10D': {'weight': 0.33, 'type': 'DoubleAttrOperator', 'value_map': CONSTANT_MAP('uniform')},
    'ts_corr_20D': {'weight': 0.33, 'type': 'DoubleAttrOperator', 'value_map': CONSTANT_MAP('uniform')},
    'ts_corr_40D': {'weight': 0.33, 'type': 'DoubleAttrOperator', 'value_map': CONSTANT_MAP('uniform')},
}).T
funcs_df['weight'] = funcs_df['weight'].astype(float)

valid_funcs = funcs_df.index.tolist()

###### Validation Check #####

# 保证待择函数均已实现
for func in valid_funcs:
    assert func in funcName2func.keys(
    ), f'funcs_df contains not implemented func: {func}.'

# 检查是否有函数未在valid_funcs中
for func in funcName2func.keys():
    if func not in valid_funcs:
        print(
            f'Warning: Implemented func {func} assigned in tain_config.py, add it into funcs_df to enable it.')
