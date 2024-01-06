# Author: Qize Liu
# Description: 操作符类型映射关系
import random


class META_ABSORBING:
    """
    吸收型映射
    """

    def get(self, x):
        return 'ticker_float'

    def __getitem__(self, x):
        return 'ticker_float'


class IDENTICAL_MAP:
    """
    恒等映射
    """

    def get(self, x):
        return x

    def __getitem__(self, x):
        return x


class CONSTANT_MAP:
    """
    常值映射
    """

    def __init__(self, value):
        self.value = value

    def get(self, x):
        return self.value

    def __getitem__(self, x):
        return self.value


class ADD_SUB_MAP:
    """
    加减法映射, 输入两列通过 ',' 连接
    """

    def get(self, x):
        col1, col2 = x.split(',')
        if 'ticker_float' in x:
            # ticker_float 吸收一切其他类型
            return 'ticker_float'
        elif col1 == col2:
            return col1
        elif 'norm' in x and 'uniform' in x:
            # norm 和 uniform 相加视为 norm
            return 'norm'
        else:
            return 'ticker_float'

    def __getitem__(self, x):
        return self.get(x)


class MUL_MAP(META_ABSORBING):
    """TODO: 乘法映射"""


class DIV_MAP(META_ABSORBING):
    """TODO: 除法映射"""
