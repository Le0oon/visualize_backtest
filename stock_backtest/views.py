from django.http import HttpResponse
from django.shortcuts import render,redirect
from django_pandas.io import read_frame
from django.contrib import messages
from django.dispatch import Signal
import pandas as pd
import json
import os
from django.conf import settings
from .backtest import main
# from celery import shared_task

evaluation_complete = Signal()



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
    'ts_ratio_beyond_3sigma': {'type': '时序单列操作符', 'description': '计算时序数据超过3倍标准差的比例'},
    'ts_ratio_beyond_2sigma': {'type': '时序单列操作符', 'description': '计算时序数据超过2倍标准差的比例'},
    'ts_index_mass_median': {'type': '时序单列操作符', 'description': '计算时序数据的中位数指标质量'},
    'ts_number_cross_mean': {'type': '时序单列操作符', 'description': '计算时序数据与均值交叉的次数'},
    'ts_time_asymmetry_stats': {'type': '时序单列操作符', 'description': '计算时序数据的时间不对称性统计量'},
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
    'ts_corr_20D': {'type': '双列操作符', 'description': '计算两个因子的20天相关系数'},
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
        'factor_list': get_factor_list(),
        }
    return render(request,'stock_backtest/write_factor.html',context)


def backtest(request):
    return render(request,'stock_backtest/backtest.html')


def get_factor_list():
    factor_list = os.listdir('./static/output')
    
    return factor_list
# from visualize_backtest

# @shared_task(name='async_evaluate_factor')
def async_evaluate_factor(start_dt, end_dt,universe,rtype,expression,turnover_fee,factor_name):
    
    main(start_dt,end_dt,
        universe=universe,
        rtype=rtype,
        expression=expression,
        turnover_fee=turnover_fee,
        factor_name=factor_name
        )

def evaluating(request):
    factor_name = request.POST.get('factor_name')
    universe = request.POST.get('universe')
    return_type = request.POST.get('return_type')
    start_date = request.POST.get('start_date')
    end_date = request.POST.get('end_date')
    factor_expression = request.POST.get('factor_expression')
    turnover_fee = float(request.POST.get('turnover_fee'))
    
    ########
    print('start_evaluation...')
    try:
        async_evaluate_factor(start_date, end_date,
                          universe, return_type, factor_expression,
                          turnover_fee, factor_name)
    except Exception as e:
        error_message = str(e)

        redirect('stock_backtest:write_factor',{'error_message':error_message})
        
    params = {
        'factor_name': factor_name,
        'universe': universe,
        'return_type': return_type,
        'start_date': start_date,
        'end_date': end_date,
        'factor_expression': factor_expression
    }
    
    # Save params to JSON file
    with open(f'./static/output/{factor_name}/params.json', 'w') as file:
        json.dump(params, file)
    ########
    print('end_evaluation')
    
    return redirect('stock_backtest:evaluation_result',
                    factor_name=factor_name,)

def info(request):
    return HttpResponse('Author: qzliu;    Email: liuqize19@gmail.com, liuqz23@mails.tsinghua.edu.cn;')

def evaluation_result(request,factor_name):

    ########
    # TODO: 等待 Backtest Frame 的输出, listen to the output
    ########
    
        
    # 这个最后要注释掉
    # return HttpResponse(factor_name)

    # 读取 factor_name 对应的文件, 画图
    ic_rst = pd.read_csv(f'./static/output/{factor_name}/ic_rst.csv').rename(columns={'Unnamed: 0':'时间'})
    ic_rst[ic_rst.columns[1:]] = ic_rst[ic_rst.columns[1:]].round(3)
    ret_rst = pd.read_csv(f'./static/output/{factor_name}/ret_rst.csv').rename(columns={'Unnamed: 0':'时间'})
    ret_rst[ret_rst.columns[1:]] = ret_rst[ret_rst.columns[1:]].round(3)
    ret_rst.dropna(inplace=True,axis=1)
    try:
        with open(f'./static/output/{factor_name}/params.json', 'r') as file:
            params = json.load(file)
    except:
        print('params.json not found')
        params = {
            'factor_name': factor_name,
            'universe': 'Unknown',
            'return_type': 'Unknown',
            'start_date': 'Unknown',
            'end_date': 'Unknown',
            'factor_expression': 'Unknown'
        }
    context = {
        'hist_path': f'output/{factor_name}/FactorHistogram.png',
        'quantile_path': f'output/{factor_name}/QuantilePlot.png',
        'auto_corr_path': f'output/{factor_name}/FactorAutoCorr.png',
        'count_path': f'output/{factor_name}/FactorStockCount.png',
        'ic_layer_path': f'output/{factor_name}/LayeredReturn.png',
        'cum_ic_path': f'output/{factor_name}/CumulativeIC.png',
        'cum_ret_path': f'output/{factor_name}/LongShortReturn.png',
        'ic_cols':ic_rst.columns.tolist(),
        'ic_data':ic_rst.values,
        'ret_cols':ret_rst.columns.tolist(),
        'ret_data':ret_rst.values,
        'factor_list': get_factor_list(),
    }
    context = {**context, **params}
    print(context)
    # TODO: 需要实现 evaluation_result.html 的内容, 把context里的数据展示到前端
    return render(request,'stock_backtest/evaluation_result.html',context)