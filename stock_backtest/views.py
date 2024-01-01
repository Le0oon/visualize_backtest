from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.contrib import messages


import json

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
    return render(request,'stock_backtest/write_factor.html')


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