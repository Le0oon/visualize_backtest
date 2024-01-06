# Base Classes for Population and Node
from abc import ABC, abstractmethod
from AutoAlpha import SingleAttrTsOperator, SingleAttrOperator, DoubleAttrOperator
import types
import random
import numpy.random as npr
import pandas as pd
import numpy as np
from AutoAlpha import train_config as config
from typing import List
import os
from multiprocessing import Pool, cpu_count
import functools
from tqdm import tqdm
import pdb
import logging

OPERATOR = 'operator'
COL_NAME = 'col_name'
INTEGER = 'integer'
N_CPU = cpu_count() // 2


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


class BaseExpressionTree(ABC):
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

    ######################################## 初始化树 ########################################

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

        BaseExpressionTree._initialize(self.root, max_depth, stop_prob)

        return self

    @staticmethod
    def _initialize(node, max_depth=None, stop_prob=None):
        layer = [node]
        for _ in range(max_depth):
            new_layer = []
            for node in layer:
                if npr.rand() < stop_prob or _ == max_depth - 1:
                    # 停止，返回列名
                    node.func = BaseExpressionTree.choose_leaf_randomly(COL_NAME)
                else:
                    node.func = BaseExpressionTree.choose_function_from_class_randomly()
                    chosen_class = node.func_class
                    if chosen_class == SingleAttrTsOperator:
                        node.child = [
                            Node(), Node(BaseExpressionTree.choose_leaf_randomly(INTEGER))]
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

    def initialize_with_expression(self, expression: str) -> Node():
        """
        根据表达式初始化
        """
        self.root = BaseExpressionTree._initialize_with_expression(expression)

        return self

    @staticmethod
    def _initialize_with_expression(expression):
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
            result = BaseExpressionTree._split_string_with_balanced_parentheses(expression)
            # print(result)
            child = [BaseExpressionTree._initialize_with_expression(s) for s in result]
            return Node(func=config.funcName2func[func_name], child=child)

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
    ######################################## 计算树表达式取值 ########################################

    def calc_tree(self, df: pd.DataFrame):
        """计算树，返回列名
        """
        origin_cols = df.columns.tolist()

        self.col_name = BaseExpressionTree._calc_tree(df, self.root)

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
            fcol = BaseExpressionTree._calc_tree(df, node.child[0])
            n = BaseExpressionTree._calc_tree(df, node.child[1])

            col_name = f"{node.func.__name__}({fcol},{n})"
            if col_name not in df.columns:
                col_name = node.func(df, fcol, n)
                df[col_name] = df[col_name].astype(np.float32)

        elif func_class_name == "SingleAttrOperator":
            # input: df, fcol
            fcol = BaseExpressionTree._calc_tree(df, node.child[0])

            col_name = f"{node.func.__name__}({fcol})"
            if col_name not in df.columns:
                col_name = node.func(df, fcol)

        elif func_class_name == "DoubleAttrOperator":
            # input: df, fcol1, fcol2
            fcol1 = BaseExpressionTree._calc_tree(df, node.child[0])
            fcol2 = BaseExpressionTree._calc_tree(df, node.child[1])

            col_name = f"{node.func.__name__}({fcol1},{fcol2})"
            if col_name not in df.columns:
                col_name = node.func(df, fcol1, fcol2)

        else:
            raise Exception(
                f'ExpressionTree._calc_tree: Invalid node function {node.func.__name__}.')

        return col_name

    ######################################## 评估树, 需重载 ########################################
    @abstractmethod
    def evaluate_tree():
        pass

    ######################################## 剪枝树至给定深度 ########################################

    def reduce_tree(self, max_depth=None):
        if max_depth is None:
            max_depth = self.max_depth * 2
        self.root = BaseExpressionTree._reduce_tree(self.root, max_depth)

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
                                    Node(BaseExpressionTree.choose_leaf_randomly(COL_NAME)))
                            else:
                                new_child.append(child)
                        node.child = new_child
                break
        return root

    ######################################## utils and properties ########################################
    @property
    def expression(self):
        return str(self.root)

    def __str__(self):
        return f"ExpressionTree(\n\t{self.root}\n)"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, BaseExpressionTree):
            return False
        else:
            return self.root == other.root

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

    def get_all_nodes(self):

        return self.root.get_all_nodes()

    def get_all_nodes_with_given_type(self, node_type=OPERATOR):
        """返回所有对应类型的节点
        """
        node_list = self.get_all_nodes()
        candidate_list = [
            node for node in node_list if node.node_type == node_type]

        return candidate_list

    @staticmethod
    def expression_depth(expression: str):
        depth = 1
        max_depth = 1
        for char in expression:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        return max_depth

    @staticmethod
    def _split_string_with_balanced_parentheses(string):
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


class BasePopulation(ABC):
    def __init__(self,
                 population_size: int,
                 max_depth: int,
                 df: pd.DataFrame,
                 epoch=10,
                 time_col='TradingDay',
                 secu_col='SecuCode',
                 indu_col='SWF',
                 ret_col='raw_evaluate',
                 # 训练参数
                 mutation_rate=1,  # 变异概率
                 crossover_rate=0.5,  # 交叉概率
                 elite_keep_rate=0.1,  # 精英保留率
                 fitness_score_thresh=7,  # 适应度分数阈值, 高过阈值的因子将进入因子库，并用于对return 计算残差
                 reduce_return=False,  # 是否对return 进行残差计算
                 # 存储参数
                 cache_max_depth=2,  # 释放缓存时，保留的因子深度
                 store_intermediate=False,  # 是否存储中间结果, 推荐不存储
                 clear_all_after_iter=True,  # 是否在每个epoch结束后清空所有中间结果
                 # 记录参数
                 save_dir=None,
                 logger=None,
                 # expression tree 类
                 expression_tree_cls=BaseExpressionTree,

                 ):
        self.df = df.reset_index(drop=True)
        self.time_col = time_col
        self.secu_col = secu_col
        self.indu_col = indu_col
        self.ret_col = ret_col
        self.population = []  # List[ExpressionTree]
        self.population_col_name = []  # List[str] 用于存储每个树的计算结构的列名
        self.expression_tree_cls = expression_tree_cls
        self.logger = logger

        # 训练参数记录
        self.population_size = population_size
        self.max_depth = max_depth
        self.epoch = epoch
        self.ret_col = ret_col
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.ndays = self.df[self.time_col].nunique()
        self.min_acceptable_factor_size = max(100, self.ndays // 5)
        self.elite_sample_size = int(self.population_size * elite_keep_rate)
        self.reduce_return = reduce_return
        self.fitness_score_thresh = fitness_score_thresh
        self.fitness_df = None

        # 存储参数
        self.cache_max_depth = cache_max_depth
        self.store_intermediate = store_intermediate
        self.clear_all_after_iter = clear_all_after_iter

        # 记录参数
        self.kept_cols = [self.time_col, self.secu_col, self.indu_col, self.ret_col] + \
            config.column_info.index.tolist()
        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def prepare_population(self, expression_list: List[str] = None):
        """初始化种群
        """
        if expression_list is not None:
            self.population = [
                self.expression_tree_cls(Node(),
                                         self.max_depth, store_intermediate=self.store_intermediate).initialize_with_expression(
                    factor) for factor in expression_list]
        else:
            self.population = [
                self.expression_tree_cls(Node(),
                                         self.max_depth, store_intermediate=self.store_intermediate).initialize()
                for _ in range(self.population_size)]

        return self.population

    def calculate_factor(self, multi_process=False, ncpu=N_CPU) -> List[str]:
        """计算种群中每个树的表达式因子值"""
        if len(self.population) == 0:
            raise Exception(
                'BasePopulation.calculate_factor: population is empty, please use `self.prepare_population` first.')
        if multi_process:
            # print("Calculating factor with multiprocessing...")
            ncpu = ncpu if ncpu <= self.population_size else self.population_size
            sub_pop_num = min(max(self.population_size // ncpu + 1, 10), 30)
            sub_population = []
            for i in range(self.population_size):
                sub = self.population[i * sub_pop_num: (i + 1) * sub_pop_num]
                if len(sub) > 0:
                    sub_population.append(sub.copy())
                else:
                    break

            _calculate_pop_tree_value = functools.partial(
                BasePopulation.calculate_pop_tree_value, df=self.df[self.kept_cols].copy())
            with Pool(ncpu) as p:
                results = list(tqdm(p.imap(_calculate_pop_tree_value, sub_population),
                                    total=len(sub_population),
                                    desc=f'Calculating factor with MultiProcessing ({ncpu=})',
                                    unit='sub_pop')
                               )
                # pdb.set_trace()
                pop_ls, cols_ls, df_ls = zip(*results)
            self.population = sum(pop_ls, [])
            self.population_col_name = sum(cols_ls, [])

            self.df = pd.concat([self.df[self.kept_cols]].copy() +
                                [_df[_col].copy() for _df, _col in zip(df_ls, cols_ls)], axis=1)

            del sub_population, pop_ls, cols_ls, df_ls

        else:
            self.population_col_name = []
            for tree in tqdm(self.population, desc='Calculating factor'):
                col_name = tree.calc_tree(self.df)

                self.population_col_name.append(col_name)

        return self.population_col_name

    @abstractmethod
    def evaluate_population(self, multi_process=False, save_path=None, ncpu=N_CPU) -> pd.DataFrame:
        """评估种群, 返回fitness_df, columns=['factor_name', 'fitness_score']
        """
        pass

    @abstractmethod
    def evolve_population(self, fitness_df) -> List[BaseExpressionTree]:
        """形成新种群
        """

    @abstractmethod
    def end_epoch(self, release_cache: int = None, reduce_return: bool = False):
        """
        每个epoch结束后调用

        """

    def fit_by_part(self):
        """训练种群, 将种群拆分成若干小种群进行计算和评估, 以减少内存占用"""

        self.population = self.prepare_population()
        for epoch in range(self.epoch):
            if self.logger is not None:
                self.logger.info(f"Epoch {epoch + 1}...")
            else:
                print('='*20)
                print(f"Epoch {epoch + 1}...")
            fitness_df_list = []
            sub_pop_num = int(np.ceil(self.population_size / config.SUB_POP_SIZE))

            # print('Calculating factor by part...')
            # pdb.set_trace()
            for i in tqdm(range(sub_pop_num),
                          desc='Calculating and Evaluate by Part',
                          unit='sub_pop'):
                sub_atp = self.__class__(
                    config.SUB_POP_SIZE, self.max_depth, self.df.copy(), ret_col=self.ret_col)
                sub_atp.population = self.population[i *
                                                     config.SUB_POP_SIZE: (i + 1) * config.SUB_POP_SIZE]
                if len(sub_atp.population) == 0:
                    break
                sub_atp.population_col_name = sub_atp.calculate_factor(
                    multi_process=True)
                fitness_df_list.append(sub_atp.evaluate_population(
                    multi_process=True, save_path=None))
                del sub_atp.df
                sub_atp.df = None
                del sub_atp

            fitness_df = pd.concat(fitness_df_list, axis=0).reset_index(
                drop=True)
            fitness_df['selection_probability'] = fitness_df['fitness_score'] / \
                fitness_df['fitness_score'].sum()
            del fitness_df_list
            self.fitness_df = fitness_df
            if self.save_dir:
                # 保存每轮fitness_df 信息
                fitness_df.to_csv(os.path.join(
                    self.save_dir, f'fitness_epoch_{epoch}.csv'), index=False)
                # 保存表现好的因子的信息
                save_cols = fitness_df[fitness_df['ic'] >
                                       self.fitness_score_thresh]['factor_name'].tolist()
                if len(save_cols) > 0:

                    self.df[['SecuCode', 'TradingDay'] + save_cols].to_csv(
                        os.path.join(self.save_dir, f'saved_factors.csv'), mode='a')
            # 展示种群信息
            report_df = fitness_df['ic'].quantile(
                [1, 0.9, 0.8, 0.2, 0.1, 0]).to_frame().T
            report_df.columns = ['ic_max', 'ic_90%',
                                 'ic_80%', 'ic_20%', 'ic_10%', 'ic_min']
            depth = [BaseExpressionTree.expression_depth(
                col) for col in self.population_col_name]
            pop_mean_abs_ic = fitness_df['ic'].abs().mean()

            if self.logger is not None:
                self.logger.info(report_df)
                self.logger.info('Population mean abs ic: %.3f' % (pop_mean_abs_ic,))
                self.logger.info('Average depth: %.3f' % (np.mean(depth),))
            else:
                display(report_df)
                print('Population mean abs ic: ', "%.3f" % pop_mean_abs_ic)
                print('Average depth: ', np.mean(depth))
                print('='*20 + '\n')

            # 按照fitness score排序
            fitness_df = fitness_df.sort_values(
                by='fitness_score', ascending=False)

            # 形成新的种群
            self.population = self.evolve_population(fitness_df)
            # 修剪过深的树
            self.population = BasePopulation.reduce_tree_depth(
                self.population)

    def fit(self, multi_process=False):
        """训练种群
        """
        if len(self.population) < self.population_size:
            self.population = self.prepare_population()
        for epoch in range(self.epoch):
            if self.logger is not None:
                self.logger.info(f"Epoch {epoch + 1}...")
            else:
                print('='*20)
                print(f"Epoch {epoch + 1}...")
            # 计算因子值
            self.population_col_name = self.calculate_factor(multi_process)
            # 评估种群, 并计算fitness score
            fitness_df = self.evaluate_population(multi_process, save_path=os.path.join(
                self.save_dir, f'fitness_epoch_{epoch}.csv') if self.save_dir else None)
            self.fitness_df = fitness_df
            if self.save_dir:
                save_cols = fitness_df[fitness_df['ic'] >
                                       self.fitness_score_thresh]['factor_name'].tolist()
                if len(save_cols) > 0:
                    cols_tobe_saved = []
                    for col in save_cols:
                        if col in self.df.columns:
                            cols_tobe_saved.append(col)
                        else:
                            warning_info = f"Column {col} not in df.columns, skipped."
                            if self.logger is not None:
                                self.logger.warning(warning_info)
                            else:
                                print(warning_info)
                    save_df = self.df[[self.time_col, self.secu_col] + save_cols]
                    save_df = save_df.loc[:, ~save_df.columns.duplicated()]
                    save_df.to_feather(os.path.join(self.save_dir, f'saved_factors_{epoch}.feather'))
            # 展示种群信息
            report_df = fitness_df['ic'].quantile(
                [1, 0.9, 0.8, 0.2, 0.1, 0]).to_frame().T
            report_df.columns = ['ic_max', 'ic_90%',
                                 'ic_80%', 'ic_20%', 'ic_10%', 'ic_min']
            depth = [BaseExpressionTree.expression_depth(
                col) for col in self.population_col_name]
            pop_mean_abs_ic = fitness_df['ic'].abs().mean()
            if self.logger is not None:
                self.logger.info(report_df)
                self.logger.info('Population mean abs ic: %.3f' % (pop_mean_abs_ic,))
                self.logger.info('Average depth: %.3f' % (np.mean(depth),))
            else:
                display(report_df)
                print('Population mean abs ic: ', "%.3f" % pop_mean_abs_ic)
                print('Average depth: ', np.mean(depth))
                print('='*20 + '\n')
            # 按照fitness score排序
            fitness_df = fitness_df.sort_values(
                by='fitness_score', ascending=False)

            # 形成新的种群
            self.population = self.evolve_population(fitness_df)

            self.end_epoch()

    @staticmethod
    def reduce_tree_depth(population: List[BaseExpressionTree], max_depth=None):
        """将树的深度降低到max_depth
        如果max_depth为None, 则设置为 2 * max_depth(population中可接受的最大深度)
        """
        for tree in population:
            tree.reduce_tree(max_depth)

        return population

    @staticmethod
    def crossoverTrees(tree1: BaseExpressionTree, tree2: BaseExpressionTree):
        """交换两棵树的子树
        Args:
            tree1 (ExpressionTree): [description]
            tree2 (ExpressionTree): [description]
        """
        node2_candidate = []
        cnt = 0
        while len(node2_candidate) == 0 and cnt < 5:
            # 随机选择一个节点，然后在另一棵树中找到同类型的节点
            node1 = random.choice(tree1.get_all_nodes())
            node2_candidate = tree2.get_all_nodes_with_given_type(node1.node_type)
            cnt += 1
        if len(node2_candidate) == 0:
            return None, None
        node2 = random.choice(node2_candidate)
        node1.child, node2.child = node2.child, node1.child
        node1.func, node2.func = node2.func, node1.func

        return tree1, tree2

    @staticmethod
    def mutateTree(tree: BaseExpressionTree):
        """随机选择一个节点，随机变异

        Args:
            tree (ExpressionTree): [description]
        """
        if tree is None:
            return None
        all_nodes = tree.get_all_nodes()
        node = random.choice(all_nodes)
        if node.node_type == OPERATOR:
            node.func = BaseExpressionTree.choose_function_from_class_randomly(
                node.func_class.__name__)
        elif node.node_type == COL_NAME:
            # TODO: comparable column choice
            node.func = BaseExpressionTree.choose_leaf_randomly(COL_NAME)
        elif node.node_type == INTEGER:
            node.func = BaseExpressionTree.choose_leaf_randomly(INTEGER)
        else:
            raise Exception('Invalid node type')

        return tree

    @staticmethod
    def sample_from_population(
            population: list, fitness_df: pd.DataFrame, sample_size=2) -> List[BaseExpressionTree]:
        """从种群中随机选择个体, 注意保证population index 与 fitness_df index 一致
        fitness_df.columns = ['fitness_score']
        """
        if 'selection_probability' not in fitness_df.columns:
            # 选择概率
            fitness_df['selection_probability'] = fitness_df['fitness_score'] / \
                fitness_df['fitness_score'].sum()
        fitness_df['selection_probability'] = fitness_df['selection_probability'].fillna(0)

        # 抽样
        sample = npr.choice(
            fitness_df.index,
            size=sample_size,
            p=fitness_df['selection_probability'].values,
            replace=False)

        return [population[i] for i in sample]

    @staticmethod
    def calculate_pop_tree_value(population: List[BaseExpressionTree], df: pd.DataFrame):
        """计算树的值, population 中df应该为同一个
        """
        df = df.copy()
        col_name = [tree.calc_tree(df) for tree in population]

        return population, col_name, df
