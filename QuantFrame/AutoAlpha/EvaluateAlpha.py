from typing import List

from .BaseClasses import BasePopulation, BaseExpressionTree
from StockBackTest import BackTest
import pandas as pd
from glob import glob
import os
import logging

logger = logging.getLogger(__name__)


class EvaluateAlpha:
    """

    properties: 
        - factor_names,
        - factor_df,
        - rst_df
    """

    def __init__(self,
                 rst_path: str,
                 df: pd.DataFrame,
                 start_date: str,  # 回测开始日期, 因为部分因子计算需使用rolling值, 减少na值出现
                 ret_col: str,
                 min_valid_propotion=0.6,
                 time_col='TradingDay',
                 secu_col='SecuCode',
                 indu_col='SWF',
                 logger=logger):
        # 保存参数
        self.logger = logger
        self.min_valid_propotion = min_valid_propotion
        self.df = df
        self.start_date = start_date
        self.ret_col = ret_col
        self.time_col = time_col
        self.secu_col = secu_col
        self.indu_col = indu_col

        # 读取数据
        fitness_files = glob(rst_path + '/*.csv')
        self.fitness_files = [f for f in fitness_files if 'warm_start_up' not in f]
        population_files = glob(rst_path + '/*.feather')
        self.population_files = [f for f in population_files if 'warm_start_up' not in f]

        fitness_df = pd.concat([pd.read_csv(f) for f in self.fitness_files])

        # 处理可接受最小样本量:
        min_valid_size = int(fitness_df['valid_factor_size'].max() * self.min_valid_propotion)

        fitness_df.reset_index(drop=True, inplace=True)
        fitness_df = fitness_df.query(
            f'valid_factor_size > {min_valid_size}').sort_values('ic', ascending=False, key=abs)
        fitness_df.drop_duplicates(subset=['factor_name'], inplace=True)
        self.logger.info(f'Original population size: {fitness_df.shape[0]}')

        # 在drop 相同因子值时优先保留因子长度短的
        fitness_df['factor_depth'] = fitness_df['factor_name'].map(BaseExpressionTree.expression_depth)
        fitness_df.sort_values('factor_depth', inplace=True, ascending=True)
        fitness_df.drop_duplicates(subset=['ic', 'tvalue'], inplace=True, keep='first')
        self.logger.info(f'After drop duplicates: {fitness_df.shape[0]}')
        fitness_df.query('abs(tvalue) > 3', inplace=True)
        self.logger.info(f'After drop abs(tvalue) < 3: {fitness_df.shape[0]}')

        # 存储因子名称
        self.factors_names = fitness_df['factor_name'].unique().tolist()
        self.train_rst_df = fitness_df
        # print(len(self.factors_names))
        # print(len(fitness_df))

    def evaluate_population(self):
        valid_pop = EvalPopulation(self.df, len(self.factors_names), self.time_col,
                                   self.secu_col, self.indu_col, self.ret_col)
        valid_pop.population = valid_pop.prepare_population(self.factors_names)
        self.logger.info(f"Start calculating factor with length {len(valid_pop.population)}")
        # return valid_pop
        # print(len(valid_pop.population))

        self.factors_names = valid_pop.calculate_factor(True)
        # print(valid_pop.df.shape)
        factor_df = valid_pop.df
        factor_df.query(f'{self.time_col} >= "{self.start_date}"', inplace=True)
        # print(factor_df.shape)
        valid_rst = BackTest.batch_ic_analysis(factor_df, rtype=self.ret_col, plot=False, ic_thresh=1)
        rst_df = valid_rst['ic_rst'].sort_values('IC(%)', ascending=False, key=abs)

        self.factor_df = factor_df
        self.rst_df = rst_df

        return rst_df


class EvalExpressionTree(BaseExpressionTree):
    def evaluate_tree():
        pass


class EvalPopulation(BasePopulation):

    def __init__(self, df: pd.DataFrame,
                 population_size: int,
                 time_col='TradingDay',
                 secu_col='SecuCode',
                 indu_col='SWF',
                 ret_col='raw_evaluate',
                 ):
        super().__init__(population_size=population_size, max_depth=0,
                         time_col=time_col, secu_col=secu_col, indu_col=indu_col, ret_col=ret_col,
                         df=df, expression_tree_cls=EvalExpressionTree)

    def evaluate_population(self):
        pass

    def evolve_population(self, fitness_df):
        pass

    def end_epoch(self):
        pass
