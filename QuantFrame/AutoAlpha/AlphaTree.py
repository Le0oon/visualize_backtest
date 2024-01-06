# Author: Qize Liu
# Description: Node and Tree structure for AlphaTree

from AutoAlpha import SingleAttrTsOperator, SingleAttrOperator, DoubleAttrOperator
import types
from Factors import NumbaFuncs
import random
import numpy.random as npr
import pandas as pd
import numpy as np
from AutoAlpha import train_config as config
import re

OPERATOR = 'operator'
COL_NAME = 'col_name'
INTEGER = 'integer'

DEBUG_MODE = False


class Node:
    def __init__(self, func=None, child=None):
        """
        initialize node.
        node_type: one of ['operator', 'col_name', 'int'].
        Args:
            func (None): Operators static method (function) or column name (str) or integer (int)

        """

        self._func = None
        self.node_type = None
        self.func_class = None  # 当func 为OPERATOR 时对应的类 type: class
        self.func = func
        self.child = child if child is not None else []

    def get_all_nodes(self):
        """返回所有节点
        """
        node_list = [self]
        for node in self.child:
            node_list.extend(node.get_all_nodes())

        return node_list

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, value):
        self._func = value
        self.node_type = OPERATOR if isinstance(value, types.FunctionType) else\
            COL_NAME if isinstance(value, str) else INTEGER if isinstance(value, (int, np.integer))\
            else None
        if self.node_type == OPERATOR:
            func_class_name = config.funcs_df.loc[self._func.__name__, 'type']
            self.func_class = SingleAttrOperator if func_class_name == 'SingleAttrOperator' else\
                SingleAttrTsOperator if func_class_name == 'SingleAttrTsOperator' else\
                DoubleAttrOperator if func_class_name == 'DoubleAttrOperator' else None
        else:
            self.func_class = None

    def __repr__(self):

        return self.__str__()

    def __str__(self):
        child_repr = ','.join([repr(child) for child in self.child])
        if self.node_type == OPERATOR:
            return f"{self.func.__name__}({child_repr})"
        elif self.node_type == COL_NAME:
            return self.func
        elif self.node_type == INTEGER:
            return str(self.func)
        elif self.node_type is None:
            return 'None'
        else:
            raise Exception('Invalid Node')

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        else:
            return self.__str__() == other.__str__()


class ExpressionTree:
    def __init__(self,
                 root: Node = Node(),
                 max_depth=5,  # 树最大深度
                 stop_prob=None,  # 分层时停止概率
                 store_intermediate=False  # 计算完毕后是否删除中间列
                 ):
        self.root = root
        self.col_name = None
        self.max_depth = max_depth
        self.stop_prob = stop_prob
        self.store_intermediate = store_intermediate

    @property
    def expression(self):
        return str(self.root)

    def __str__(self):
        return f"ExpressionTree(\n\t{self.root}\n)"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, ExpressionTree):
            return False
        else:
            return self.root == other.root

    def get_all_nodes(self):

        return self.root.get_all_nodes()

    def get_all_nodes_with_given_type(self, node_type=OPERATOR):
        """返回所有对应类型的节点
        """
        node_list = self.get_all_nodes()
        candidate_list = [
            node for node in node_list if node.node_type == node_type]

        return candidate_list

    def calc_tree(self, df: pd.DataFrame):
        """计算树，返回列名
        """
        origin_cols = df.columns.tolist()

        self.col_name = ExpressionTree._calc_tree(df, self.root)

        if not self.store_intermediate:
            df.drop(columns=[col for col in df.columns if (col not in origin_cols) and (col != self.col_name)],
                    inplace=True, errors='ignore')

        return self.col_name

    @staticmethod
    def _calc_tree(df, node: Node):
        if node.node_type in [INTEGER, COL_NAME]:
            # 叶子节点, 保留
            return node.func

        func_class_name = config.funcName2className.get(node.func.__name__)
        if func_class_name == "SingleAttrTsOperator":
            # input: df, fcol, n
            fcol = ExpressionTree._calc_tree(df, node.child[0])
            n = ExpressionTree._calc_tree(df, node.child[1])

            col_name = f"{node.func.__name__}({fcol},{n})"
            if col_name not in df.columns:
                col_name = node.func(df, fcol, n)
                df[col_name] = df[col_name].astype(np.float32)

        elif func_class_name == "SingleAttrOperator":
            # input: df, fcol
            fcol = ExpressionTree._calc_tree(df, node.child[0])

            col_name = f"{node.func.__name__}({fcol})"
            if col_name not in df.columns:
                col_name = node.func(df, fcol)

        elif func_class_name == "DoubleAttrOperator":
            # input: df, fcol1, fcol2
            fcol1 = ExpressionTree._calc_tree(df, node.child[0])
            fcol2 = ExpressionTree._calc_tree(df, node.child[1])

            col_name = f"{node.func.__name__}({fcol1},{fcol2})"
            if col_name not in df.columns:
                col_name = node.func(df, fcol1, fcol2)

        else:
            raise Exception(
                f'ExpressionTree._calc_tree: Invalid node function {node.func.__name__}.')

        return col_name

    @property
    def depth(self):
        """depth of tree
        """
        depth = 0
        node_list = [self.root]
        while len(node_list) > 0:
            depth += 1
            new_list = []
            for node in node_list:
                new_list = new_list + node.child
            node_list = new_list

        return depth

    def initialize(self):
        """初始化Expression Tree"""
        if self.max_depth:
            max_depth = self.max_depth
            stop_prob = 0
        elif self.stop_prob:
            # 作为braching process 考虑，mu = 4/3 * (stop_prob)
            max_depth = 20
            stop_prob = self.stop_prob
        else:
            raise Exception(
                'ExpressionTree.initialize: either max_depth or stop_prob should be specified.')

        ExpressionTree._initialize(self.root, max_depth, stop_prob)

        return self

    def initialize_with_expression(self, expression: str) -> Node():
        """
        根据表达式初始化
        """
        self.root = ExpressionTree.parse_expression(expression)

        return self

    @staticmethod
    def parse_expression(expression):
        # print(expression)
        func_name = ''
        for i, s in enumerate(expression):
            if s == '(':
                expression = expression[:-1]
                break
            else:
                func_name += s
        if i == len(expression)-1:
            return Node(int(func_name) if func_name.isdigit() else func_name)
        else:
            expression = expression[i+1:]
            result = split_string_with_balanced_parentheses(expression)
            # print(result)
            child = [ExpressionTree.parse_expression(s) for s in result]
            return Node(func=config.funcName2func[func_name], child=child)

    @staticmethod
    def _initialize(node, max_depth=None, stop_prob=None):
        layer = [node]
        for _ in range(max_depth):
            new_layer = []
            for node in layer:
                if npr.rand() < stop_prob or _ == max_depth - 1:
                    # 停止，返回列名
                    node.func = ExpressionTree.choose_leaf_randomly(COL_NAME)
                else:
                    node.func = ExpressionTree.choose_function_from_class_randomly()
                    chosen_class = node.func_class
                    if chosen_class == SingleAttrTsOperator:
                        node.child = [
                            Node(), Node(ExpressionTree.choose_leaf_randomly(INTEGER))]
                        new_layer.append(node.child[0])
                    elif chosen_class == SingleAttrOperator:
                        node.child = [Node()]
                        new_layer.append(node.child[0])
                    elif chosen_class == DoubleAttrOperator:
                        node.child = [Node(), Node()]
                        new_layer.append(node.child[0])
                        new_layer.append(node.child[1])
                    else:
                        raise Exception(
                            f'ExpressionTree._initialize: Invalid node function {node.func.__name__}.')

            if len(new_layer) == 0:
                break

            layer = new_layer

    def reduce_tree(self, max_depth=None):
        if max_depth is None:
            max_depth = self.max_depth * 2
        self.root = ExpressionTree._reduce_tree(self.root, max_depth)

        return self

    @staticmethod
    def _reduce_tree(root: Node, max_depth=10):
        depth = 1
        node_list = [root]
        while len(node_list) > 0:
            depth += 1
            new_list = []
            for node in node_list:
                new_list = new_list + node.child
            node_list = new_list
            if depth == max_depth - 1:
                for node in node_list:
                    if node.node_type == OPERATOR:
                        new_child = []
                        for child in node.child:
                            if child.node_type == OPERATOR:
                                new_child.append(
                                    Node(ExpressionTree.choose_leaf_randomly(COL_NAME)))
                            else:
                                new_child.append(child)
                        node.child = new_child
                        # print('==Root==', node, node.node_type)
                        # print(child, new_child)

                break
        return root

    def evaluate_tree(self, df, ret_col='raw_evaluate', ret_normalized=True):
        """
        return向后shift两天
        df['raw_evaluate'] = df.groupby('SecuCode')['raw_close_close'].shift(-2)

        return IC(%), tvalue, number of sample
        """
        if not ret_normalized:
            df[ret_col] = df.groupby('TradingDay')[ret_col].transform(
                lambda x: NumbaFuncs.normalize(x.values, False))
        factor_col = self.col_name
        df[f'normalized_{factor_col}'] = df.groupby(
            'TradingDay')[factor_col].transform(lambda x: NumbaFuncs.normalize(x.values, True))
        df[f'ic_{factor_col}'] = df[f'normalized_{factor_col}'] * df[ret_col]
        ic_ts = df.groupby('TradingDay')[f'ic_{factor_col}'].mean().dropna()
        df.drop(columns=[
            f'ic_{factor_col}', f'normalized_{factor_col}'], inplace=True, errors='ignore')
        # print(ic_ts)
        if len(ic_ts) == 0:
            return 0, 0, 0

        return ic_ts.mean() * 100, NumbaFuncs.tvalue(ic_ts.values), len(ic_ts)

    @staticmethod
    def choose_function_from_class_randomly(func_class=None):
        """随机返回对应类的函数, 用于初始化
        若func_class 为None, 则按权重随机返回所有函数; 否则返回对应类的函数

        Args:
            func_type (_type_): class
        """
        if func_class is not None:
            df = config.funcs_df.query(f'type == "{func_class}"').copy()
        else:
            df = config.funcs_df.copy()

        df['weight'] = df['weight']/df['weight'].sum()
        chosen_func_name = npr.choice(df.index, p=df['weight'].values)

        return config.funcName2func[chosen_func_name]

    @staticmethod
    def choose_leaf_randomly(node_type=COL_NAME):
        """随机返回对应类型的叶子节点, 用于初始化

        Args:
            node_type (_type_, optional): either of COL_NAME or INTEGER.
        """

        if node_type == COL_NAME:
            df = config.column_info.copy()
            df['weight'] = df['weight']/df['weight'].sum()

            return npr.choice(df.index, p=df['weight'].values)

        elif node_type == INTEGER:
            df = config.integer_info.copy()
            df['weight'] = df['weight']/df['weight'].sum()

            return npr.choice(df.index, p=df['weight'].values)

        else:
            raise Exception(f'ExpressionTree.choose_leaf_randomly: Invalid node type: {node_type},\
                must be either of INTEGER or COL_NAME.')


def calculate_expression_depth(expression: str):
    depth = 1
    max_depth = 1
    for char in expression:
        if char == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ')':
            depth -= 1
    return max_depth


def split_string_with_balanced_parentheses(string):
    child = []
    sub_string = string
    paired = 0
    idx = 0
    # cnt = 0
    while len(sub_string) > 0:
        # print(idx, sub_string)
        if idx == len(sub_string) - 1:
            child.append(sub_string)
            break
        if sub_string[idx] == '(':
            paired += 1
            idx += 1
        elif sub_string[idx] == ')':
            paired -= 1
            idx += 1
        elif sub_string[idx] == ',' and paired == 0:
            child.append(sub_string[:idx])
            sub_string = sub_string[idx+1:]
            idx = 0
        else:
            idx += 1
        # cnt += 1

    # print(cnt)
    return child
