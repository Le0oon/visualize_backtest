# Autor: Qize Liu
# Description: Population manipulation functions

from .BaseClasses import BasePopulation, BaseExpressionTree
from Factors import NumbaFuncs
import numpy.random as npr
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List
import functools
from AutoAlpha import train_config as config
N_CPU = cpu_count() // 2


class NaiveExpressionTree(BaseExpressionTree):
    def evaluate_tree(self, df, ret_col='raw_evaluate', ret_normalized=True):

        return NaiveExpressionTree.evaluate_tree_static(df, self.col_name, ret_col, ret_normalized)

    @staticmethod
    def evaluate_tree_static(df, factor_col, ret_col, date_col='TradingDay', ret_normalized=True):
        """
        return向后shift两天
        df['raw_evaluate'] = df.groupby('SecuCode')['raw_close_close'].shift(-2)

        return IC(%), tvalue, number of sample
        """

        if not ret_normalized:
            df[ret_col] = df.groupby(date_col)[ret_col].transform(
                lambda x: NumbaFuncs.normalize(x.values, False))
        df[f'normalized_{factor_col}'] = df.groupby(
            date_col)[factor_col].transform(lambda x: NumbaFuncs.normalize(x.values, True))
        df[f'ic_{factor_col}'] = df[f'normalized_{factor_col}'] * df[ret_col]
        ic_ts = df.groupby(date_col)[f'ic_{factor_col}'].mean().dropna()
        df.drop(columns=[
            f'ic_{factor_col}', f'normalized_{factor_col}'], inplace=True, errors='ignore')
        # print(ic_ts)
        if len(ic_ts) == 0:
            return 0, 0, 0

        return ic_ts.mean() * 100, NumbaFuncs.tvalue(ic_ts.values, avoid_zero_div=True), len(ic_ts)

    def __str__(self):
        return f"NaiveExpressionTree(\n\t{self.root}\n)"


class NaivePopulation(BasePopulation):
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
                 ###### customized_params ######
                 corr_thresh=None,
                 warm_start=None,
                 ):
        super().__init__(population_size, max_depth, df, epoch, time_col, secu_col, indu_col, ret_col,
                         mutation_rate, crossover_rate, elite_keep_rate, fitness_score_thresh, reduce_return,
                         cache_max_depth, store_intermediate, clear_all_after_iter, save_dir, logger,
                         expression_tree_cls=NaiveExpressionTree)

    def evaluate_population(self, multi_process=False, save_path=None, ncpu=N_CPU):
        """评估种群
        """
        if len(self.population_col_name) == 0:
            raise Exception(
                "Population is empty, please calculate factor first.")

        if multi_process:
            ncpu = ncpu if ncpu <= self.population_size else self.population_size
            sub_pop_num = min(max(self.population_size // ncpu + 1, 10), 30)
            sub_population = []
            for i in range(self.population_size):
                sub = self.population[i * sub_pop_num: (i + 1) * sub_pop_num]
                sub_col_name = [tree.col_name for tree in sub]
                if len(sub) > 0:
                    tmp_df = self.df[[self.time_col, self.ret_col] + sub_col_name]
                    sub_population.append(
                        tmp_df.loc[:, ~tmp_df.columns.duplicated()].copy()
                    )
                else:
                    break
            # print(sub_pop_num, len(sub_population))
            _evaluate_pop_tree = functools.partial(
                evauate_tree_wrapped,
                date_col=self.time_col,
                ret_col=self.ret_col,
                ret_normalized=True)
            # self.sub_population = sub_population
            with Pool(ncpu) as p:
                results = list(tqdm(p.imap(_evaluate_pop_tree, sub_population),
                                    total=len(sub_population),
                                    desc='Evaluating factor with MultiProcessing',
                                    unit='sub_pop')
                               )
                ic_ls, tvalue_ls, valid_factor_size_ls = zip(*results)
            self.ic_fitness = sum(ic_ls, [])
            self.tvalue_fitness = sum(tvalue_ls, [])
            self.valid_factor_size = sum(valid_factor_size_ls, [])

        else:
            # print("Evaluating population...")
            df = self.df.loc[:, ~self.df.columns.duplicated()]
            self.ic_fitness, self.tvalue_fitness, self.valid_factor_size = zip(
                *
                [tree.evaluate_tree(df, self.ret_col)
                 for tree in tqdm(self.population, desc='Evaluating factor')])

        return formuate_fitness_df(
            self.population_col_name, self.ic_fitness,
            self.tvalue_fitness, self.valid_factor_size,
            self.min_acceptable_factor_size, save_path=save_path)

    def evolve_population(self, fitness_df):
        """形成新种群
        """
        elite_sample_index = fitness_df.head(
            self.elite_sample_size).index.values
        new_sample = [self.population[i] for i in elite_sample_index]
        while len(new_sample) < self.population_size:
            sample1, sample2 = super().sample_from_population(
                self.population, fitness_df)
            if npr.rand() < self.crossover_rate:
                sample1, sample2 = super().crossoverTrees(sample1, sample2)
            if npr.rand() < self.mutation_rate:
                sample1 = super().mutateTree(sample1)
                sample2 = super().mutateTree(sample2)
            if sample1 and (str(sample1.root) not in self.df.columns) and sample1 not in new_sample:
                new_sample.append(sample1)
            if sample2 and str(sample2.root) not in self.df.columns and sample2 not in new_sample:
                new_sample.append(sample2)

        return new_sample[:self.population_size]

    def end_epoch(self, release_cache: int = None, reduce_return: bool = False):
        """
        每个epoch结束后的运行:
            - 将深度大于一定界限的树进行剪枝;
            - 释放df中深度大于5的因子列;
            - 计算return 的residual;

        """
        if self.clear_all_after_iter:

            self.df = self.df[self.kept_cols].copy()

        elif release_cache is not None:
            self.df = BasePopulation.release_cache(self.df, max_depth=release_cache,
                                                   reserved_cols=self.population_col_name)
        else:
            self.df = BasePopulation.release_cache(self.df, max_depth=self.cache_max_depth,
                                                   reserved_cols=self.population_col_name)
        self.population = BasePopulation.reduce_tree_depth(
            self.population)
        numeric_col = self.df.select_dtypes(include=np.number).columns.tolist()
        self.df[numeric_col] = self.df[numeric_col].astype(np.float32)

        if self.reduce_return:
            raise NotImplementedError()

    @staticmethod
    def release_cache(df: pd.DataFrame, max_depth=10, reserved_cols=[]):
        drop_col_name = [
            col for col in df.columns
            if col not in reserved_cols and BaseExpressionTree.expression_depth(col) > max_depth]
        df.drop(columns=drop_col_name, inplace=True)

        return df


def formuate_fitness_df(
        population_col_name: list, ic_fitness: list, tvalue_fitness: list, valid_factor_size: list,
        min_acceptable_factor_size: int,
        save_path=None):
    """将种群的fitness信息转化为DataFrame
    return_col: ['factor_name', 'ic', 'tvalue', 'valid_factor_size', 'fitness_score', 'selection_probability', 'cumulative_probability']
    """
    fitness_df = pd.DataFrame(
        columns=['factor_name', 'ic', 'tvalue', 'valid_factor_size'])
    fitness_df['factor_name'] = pd.Series(population_col_name)
    fitness_df['ic'] = pd.Series(ic_fitness)
    fitness_df['tvalue'] = pd.Series(tvalue_fitness)
    fitness_df['tvalue'] = fitness_df['tvalue'].fillna(0)
    fitness_df['valid_factor_size'] = pd.Series(valid_factor_size)
    fitness_df.reset_index(drop=True, inplace=True)

    # fitness_score = ic + tanh(tvalue/3) * 3
    fitness_df['fitness_score'] = fitness_df['ic'].abs() + (np.tanh(fitness_df['tvalue'].abs()/3) * 3)
    # 有效因子个数小于阈值的，fitness_score置为0
    fitness_df['fitness_score'] = fitness_df['fitness_score'].mask(
        fitness_df['valid_factor_size'] < min_acceptable_factor_size, 0)
    # 选择概率
    fitness_df['selection_probability'] = fitness_df['fitness_score'] / \
        fitness_df['fitness_score'].sum()
    # 保存
    if save_path:
        fitness_df.to_csv(save_path, index=False)

    return fitness_df


def evauate_tree_wrapped(df, ret_col, date_col='TradingDay', ret_normalized=True):
    """
    df: 'TradingDay', 'raw_evaluate', factor_cols
    """
    factor_cols = df.columns[2:]
    if not ret_normalized:
        df[ret_col] = df.groupby(date_col)[ret_col].transform(
            lambda x: NumbaFuncs.normalize(x.values, False))
    ic, tvalue, valid_factor_size = zip(
        *[NaiveExpressionTree.evaluate_tree_static(df, col, ret_col, date_col, ret_normalized)
          for col in factor_cols])

    return list(ic), list(tvalue), list(valid_factor_size)
