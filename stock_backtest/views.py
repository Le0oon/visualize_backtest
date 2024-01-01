from django.http import HttpResponse
from django.shortcuts import render,redirect
from django_pandas.io import read_frame

from django.contrib import messages
import pandas as pd

import json




funcs_df = pd.DataFrame({
    # 单变量时序操作符
    'ts_mean': {'type': '时序单列操作符', 'description': '计算时序数据的均值'},
    'ts_std': {'type': '时序单列操作符', 'description': '计算时序数据的标准差'},
    'ts_max': {'type': '时序单列操作符', 'description': '计算时序数据的最大值'},
    'ts_min': {'type': '时序单列操作符', 'description': '计算时序数据的最小值'},
    'ts_median': {'type': '时序单列操作符', 'description': '计算时序数据的中位数'},
    'ts_skew': {'type': '时序单列操作符', 'description': '计算时序数据的偏度'},
    'ts_kurt': {'type': '时序单列操作符', 'description': '计算时序数据的峰度'},
    'ts_autocorr': {'type': '时序单列操作符', 'description': '计算时序数据的自相关系数'},
    'ts_shift': {'type': '时序单列操作符', 'description': '对时序数据进行平移操作'},
    'ts_diff': {'type': '时序单列操作符', 'description': '计算时序数据的差分'},
    'ts_median_abs_deviation': {'type': '时序单列操作符', 'description': '计算时序数据的中位数绝对偏差'},
    'ts_rms': {'type': '时序单列操作符', 'description': '计算时序数据的均方根'},
    'ts_norm_mean': {'type': '时序单列操作符', 'description': '计算时序数据的均值归一化'},
    'ts_norm_min': {'type': '时序单列操作符', 'description': '计算时序数据的最小值归一化'},
    'ts_norm_max': {'type': '时序单列操作符', 'description': '计算时序数据的最大值归一化'},
    'ts_norm_min_max': {'type': '时序单列操作符', 'description': '计算时序数据的最小-最大值归一化'},
    'ts_ratio_beyond_3sigma': {'type': '时序单列操作符', 'description': '计算时序数据超过3倍标准差的比例'},
    'ts_ratio_beyond_2sigma': {'type': '时序单列操作符', 'description': '计算时序数据超过2倍标准差的比例'},
    'ts_index_mass_median': {'type': '时序单列操作符', 'description': '计算时序数据的中位数指标质量'},
    'ts_number_cross_mean': {'type': '时序单列操作符', 'description': '计算时序数据与均值交叉的次数'},
    'ts_time_asymmetry_stats': {'type': '时序单列操作符', 'description': '计算时序数据的时间不对称性统计量'},
    'ts_longest_strike_above_mean': {'type': '时序单列操作符', 'description': '计算时序数据超过均值的最长连续长度'},
    'ts_longest_strike_below_mean': {'type': '时序单列操作符', 'description': '计算时序数据低于均值的最长连续长度'},
    'ts_mean_over_1norm': {'type': '时序单列操作符', 'description': '计算时序数据的均值除以1范数'},
    'ts_norm': {'type': '时序单列操作符', 'description': '计算时序数据的归一化'},

    # 单变量操作符
    'cs_rank': {'type': '单列操作符', 'description': '截面排序'},
    'cs_norm': {'type': '单列操作符', 'description': '截面标准化'},

    # 双变量操作符
    'comb_add': {'type': '双列操作符', 'description': '两个因子相加'},
    'comb_sub': {'type': '双列操作符', 'description': '两个因子相减'},
    'comb_mul': {'type': '双列操作符', 'description': '两个因子相乘'},
    'comb_div': {'type': '双列操作符', 'description': '两个因子相除'},
    'ts_corr_10D': {'type': '双列操作符', 'description': '计算两个因子的10天相关系数'},
    'ts_corr_20D': {'type': '双列操作符', 'description': '计算两个因子的20天相关系数'},
    'ts_corr_40D': {'type': '双列操作符', 'description': '计算两个因子的40天相关系数'},
}).T.reset_index()
funcs_df.columns = ['操作符名称','操作符种类', '操作符描述']

column_info = pd.DataFrame({
    'open': {'description': '开盘价'},
    'high': {'description': '最高价'},
    'low': {'description': '最低价'},
    'close': {'description': '收盘价'},
    'TurnoverVolume': {'description': '成交量'},
    'TurnoverValue': {'description': '成交额'},
    'vwap': {'description': '加权平均价'},
    'raw_close_close': {'description': '收盘价相对于前一天收盘价的变化'},
    'raw_close_open': {'description': '收盘价相对于当天开盘价的变化'},
    'raw_close_high': {'description': '收盘价相对于当天最高价的变化'},
    'raw_close_low': {'description': '收盘价相对于当天最低价的变化'},
    'raw_close_vwap': {'description': '收盘价相对于当天加权平均价的变化'},
}).T.reset_index()
column_info.columns = ['列名称','列描述']

# Create your views here.
def login(request):
    return render(request,'stock_backtest/login.html')


def reg(request):

    if request.method == 'POST':
        
        usr_name = request.POST.get('usr_name')
        password = request.POST.get('password')
        with open('./valid_users.json', 'r') as file:
            json_data = file.read()
        my_dict = json.loads(json_data)

        if usr_name in my_dict.keys():
            print(my_dict[usr_name],password)
            if password == my_dict[usr_name]:
                print('Correct password!!!')
                return redirect('stock_backtest:write_factor')
            else:
                print('Wrong password!!!')
        else:

            print('Wrong user name!!!')
                    
    return redirect('stock_backtest:login')

def write_factor(request):
    context = {
        # 'data': funcs_df.to_json(orient='values',force_ascii=False),
        'func_data': funcs_df.values,
        'func_columns': funcs_df.columns.tolist(),
        'col_data': column_info.values,
        'col_columns': column_info.columns.tolist(),
        
        }
    return render(request,'stock_backtest/write_factor.html',context)


def backtest(request):
    return render(request,'stock_backtest/backtest.html')

def evaluating(request):
    factor_name = request.POST.get('factor_name')
    universe = request.POST.get('universe')
    return_type = request.POST.get('return_type')
    start_date = request.POST.get('start_date')
    end_date = request.POST.get('end_date')
    factor_expression = request.POST.get('factor_expression')
    params = {
        'factor_name':factor_name,
        'universe':universe,
        'return_type':return_type,
        'start_date':start_date,
        'end_date':end_date,
        'factor_expression':factor_expression}
    ########
    # TODO: 在这里调用 Backtest Frame, 然后画图存到本地
    ########
    
    return redirect('stock_backtest:evaluation_result',
                    factor_name=factor_name,)


def evaluation_result(request,factor_name):

    ########
    # TODO: 等待 Backtest Frame 的输出, listen to the output
    ########
    
    
    # 这个最后要注释掉
    return HttpResponse(factor_name)

    # 读取 factor_name 对应的文件, 画图
    context = {}
    # TODO: 需要实现 evaluation_result.html 的内容, 把context里的数据展示到前端
    return render(request,'stock_backtest/evaluation_result.html',context)