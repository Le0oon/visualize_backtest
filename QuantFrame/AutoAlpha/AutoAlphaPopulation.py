from .BaseClasses import BasePopulation, BaseExpressionTree, Node, N_CPU
from .NaivePopulation import NaivePopulation
from AutoAlpha import train_config as config
from Factors import NumbaFuncs
import numpy.random as npr
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List
import functools
import pdb
import logging
import os
from MLMethods import CorrelationFilter


class AutoAlphaExpressionTree(BaseExpressionTree):

    def evaluate_tree(self, df, ret_col='raw_evaluate', time_col='TradingDay', secu_col='SecuCode', ret_normalized=True):

        return AutoAlphaExpressionTree.evaluate_tree_static(
            df, self.col_name, ret_col, time_col, secu_col, ret_normalized)

    @staticmethod
    def evaluate_tree_static(
            df, factor_col, ret_col, time_col='TradingDay', secu_col='SecuCode', ret_normalized=True, logger=None):
        if not ret_normalized:
            df[ret_col] = df.groupby(time_col)[ret_col].transform(
                lambda x: NumbaFuncs.normalize(x.values, False))

        df[f'normalized_{factor_col}'] = df.groupby(
            time_col)[factor_col].transform(lambda x: NumbaFuncs.normalize(x.values, True))
        df[f'ic_{factor_col}'] = df[f'normalized_{factor_col}'] * df[ret_col]
        ic_ts = df.groupby(time_col)[f'ic_{factor_col}'].mean().dropna()
        # df.drop(columns=[
        #     f'ic_{factor_col}', f'normalized_{factor_col}'], inplace=True, errors='ignore')
        first_pc = first_principal_component(df, factor_col, time_col, secu_col, logger)
        first_pc = pd.Series(first_pc, name=factor_col)
        if len(ic_ts) == 0:
            # print(f"{factor_col} has no valid ic_ts")
            # if logger:
            #     logger.warning(f"{factor_col} has no valid ic_ts")
            return 0, 0, 0, first_pc

        return ic_ts.mean() * 100, NumbaFuncs.tvalue(ic_ts.values, avoid_zero_div=True), len(ic_ts), first_pc

    def __str__(self):
        return f"AutoAlphaExpressionTree(\n\t{self.root}\n)"


class AutoAlphaPopulation(BasePopulation):
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
                 corr_thresh=0.9,
                 warm_start=True,  # bool or warm start file path

                 ):
        super().__init__(population_size, max_depth, df, epoch, time_col, secu_col, indu_col, ret_col,
                         mutation_rate, crossover_rate, elite_keep_rate, fitness_score_thresh, reduce_return,
                         cache_max_depth, store_intermediate, clear_all_after_iter, save_dir, logger,
                         expression_tree_cls=AutoAlphaExpressionTree)

        self.corr_thresh = corr_thresh
        self.warm_start = warm_start
        self.fpc = []

    def prepare_population(self, expression_list: List[str] = None):
        """增加 warm start-up 功能"""
        # print(self.expression_tree_cls)
        if self.warm_start is not None and self.warm_start is not False:
            if isinstance(self.warm_start, bool):
                # 未指定warm start population 路径, 生成新的
                start_up_population = min(self.population_size * 10, 50000)
                self.population = [
                    self.expression_tree_cls(
                        Node(), self.max_depth, store_intermediate=self.store_intermediate).initialize()
                    for _ in range(start_up_population)]
                # warm start up
                if self.logger is not None:
                    self.logger.info(f"Warm Start up with init population size: {start_up_population}...")
                else:
                    print(f"Warm Start up with init population size: {start_up_population}...")

                fitness_df_list = []
                sub_pop_num = int(np.ceil(start_up_population / config.SUB_POP_SIZE))
                for i in tqdm(range(sub_pop_num),
                              desc='Calculating and Evaluate by Part',
                              unit='sub_pop'):
                    sub_atp = self.__class__(
                        config.SUB_POP_SIZE, self.max_depth, self.df.copy(), ret_col=self.ret_col)
                    sub_atp.population = self.population[i * config.SUB_POP_SIZE: (i + 1) * config.SUB_POP_SIZE]
                    if len(sub_atp.population) == 0:
                        break
                    sub_atp.population_col_name = sub_atp.calculate_factor(multi_process=True)
                    save_path = os.path.join(
                        self.save_dir, "warm_start_up_population.csv") if self.save_dir is not None else None
                    fitness_df_list.append(sub_atp.evaluate_population(multi_process=True, save_path=save_path))
                    del sub_atp.df
                    sub_atp.df = None
                    del sub_atp

                fitness_df = pd.concat(fitness_df_list, axis=0).reset_index(drop=True)
                fitness_df.reset_index(drop=True, inplace=True)
            else:
                if self.logger is not None:
                    self.logger.info(f"Warm Start up with given data: {self.warm_start}...")
                else:
                    print(f"Warm Start up with given data: {self.warm_start}...")
                fitness_df = pd.read_csv(self.warm_start)
                fitness_df = fitness_df[fitness_df['factor_name'] != 'factor_name']
                if self.save_dir is not None:
                    fitness_df.to_csv(os.path.join(self.save_dir, "warm_start_up_population.csv"))
                # print(fitness_df.columns)
                # raise Exception()
            # return fitness_df
            fitness_df.drop_duplicates(subset=['factor_name'], inplace=True)

            fitness_df.sort_values('fitness_score', ascending=False, inplace=True)
            kept_population = fitness_df.head(self.population_size)['factor_name'].tolist()

            # 创建初始种群
            self.population = super().prepare_population(kept_population)
        else:
            super().prepare_population(expression_list)

        return self.population

    def evaluate_population(self, multi_process=True, save_path=None, ncpu=N_CPU):
        """评估种群
        """
        if len(self.population_col_name) == 0:
            raise Exception("Population is empty, please calculate factor first.")
        if multi_process:
            ncpu = ncpu if ncpu <= self.population_size else self.population_size
            sub_pop_num = min(max(self.population_size // ncpu + 1, 10), 30)
            sub_population = []
            for i in range(self.population_size):
                sub = self.population[i * sub_pop_num: (i + 1) * sub_pop_num]
                sub_col_name = [tree.col_name for tree in sub]
                if len(sub) > 0:
                    tmp_df = self.df[[self.time_col, self.secu_col, self.ret_col] + sub_col_name]
                    sub_population.append(
                        tmp_df.loc[:, ~tmp_df.columns.duplicated()].copy()
                    )
                else:
                    break

            _evaluate_pop_tree = functools.partial(
                evauate_tree_wrapped,
                time_col=self.time_col,
                secu_col=self.secu_col,
                ret_col=self.ret_col,
                ret_normalized=True,
                logger=self.logger)
            with Pool(ncpu) as p:
                results = list(tqdm(p.imap(_evaluate_pop_tree, sub_population),
                                    total=len(sub_population),
                                    desc=f'Evaluating factor with MultiProcessing ({ncpu=})',
                                    unit='sub_pop')
                               )
                ic_ls, tvalue_ls, valid_factor_size_ls, fpc = zip(*results)
            # pdb.set_trace()
            # print(len(ic_ls[0]), len(tvalue_ls[0]), len(valid_factor_size_ls[0]), len(fpc[0]))
            self.ic_fitness = sum(ic_ls, [])
            self.tvalue_fitness = sum(tvalue_ls, [])
            self.valid_factor_size = sum(valid_factor_size_ls, [])
            self.fpc = sum(fpc, [])

        else:
            # print("Evaluating population...")
            df = self.df.loc[:, ~self.df.columns.duplicated()]
            self.ic_fitness, self.tvalue_fitness, self.valid_factor_size, self.fpc = zip(
                *[tree.evaluate_tree(df, self.ret_col) for tree in tqdm(self.population)])

        return formuate_fitness_df(
            self.population_col_name, self.ic_fitness,
            self.tvalue_fitness, self.valid_factor_size,
            self.fpc,
            self.min_acceptable_factor_size,
            corr_thresh=self.corr_thresh,
            save_path=save_path,
            logger=self.logger)

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

    def evolve_population(self, fitness_df):
        """形成新种群,
        注意: fitness_df 的index 必须对应self.population的index
        """
        # population size过大时, 生成子代效率可能很低,
        # 在抽样次数超过population_size 时, 生成子代的概率为0
        # 在抽样次数超过population_size * 5 时,停止生成子代
        retry_limit = self.population_size * 5
        boost_limit = self.population_size * 2
        try_cnt = 0

        elite_sample_index = fitness_df.head(
            self.elite_sample_size).index.values
        new_sample = [self.population[i] for i in elite_sample_index]
        while len(new_sample) < self.population_size:
            if try_cnt % 200 == 0:
                self.logger.info(f"try: {try_cnt}; new_sample size: {len(new_sample)}")
            try_cnt += 1
            sample1, sample2 = super().sample_from_population(
                self.population, fitness_df)
            if try_cnt > boost_limit:
                sample1, sample2 = super().crossoverTrees(sample1, sample2)
                sample1 = super().mutateTree(sample1)
                sample2 = super().mutateTree(sample2)
                if sample1:
                    new_sample.append(sample1)
                if sample2:
                    new_sample.append(sample2)
            else:
                if npr.rand() < self.crossover_rate:
                    sample1, sample2 = super().crossoverTrees(sample1, sample2)
                if npr.rand() < self.mutation_rate:
                    sample1 = super().mutateTree(sample1)
                    sample2 = super().mutateTree(sample2)
                if sample1 and (str(sample1.root) not in self.df.columns) and (sample1 not in new_sample):
                    new_sample.append(sample1)
                if sample2 and str(sample2.root) not in self.df.columns and (sample2 not in new_sample):
                    new_sample.append(sample2)

            if try_cnt > retry_limit and len(new_sample) >= self.population_size * 0.3:
                break

        return new_sample[:self.population_size]


def first_principal_component(df, factor_col, index='TradingDay', columns='SecuCode', logger=None):
    """返回特征的第一主成分, 长度等于日期长度"""

    pvt_df = df.pivot(index=index, columns=columns, values=factor_col)
    data = (pvt_df - pvt_df.mean()).fillna(0).values
    inf_info = np.isinf(data)
    if inf_info.any():
        data = np.where(inf_info, 0, data)
    # return test_pca(data)
    try:
        return NumbaFuncs.pca(data.T)
    except:
        # if logger:
        #     logger.debug(f"{factor_col} has no valid principal component")
        # else:
        #     logging.debug(f"{factor_col} has no valid principal component")
        return np.zeros(data.shape[0])


def evauate_tree_wrapped(df, ret_col, time_col='TradingDay', secu_col='SecuCode', ret_normalized=True, logger=None):
    """
    df: 'TradingDay', 'SecuCode','raw_evaluate', factor_cols
    """
    factor_cols = df.columns[3:]
    if not ret_normalized:
        df[ret_col] = df.groupby(time_col)[ret_col].transform(
            lambda x: NumbaFuncs.normalize(x.values, False))
    ic, tvalue, valid_factor_size, fpc = zip(
        *[AutoAlphaExpressionTree.evaluate_tree_static(df, col, ret_col, time_col, secu_col, ret_normalized, logger)
          for col in factor_cols])
    # print(factor_cols)
    return list(ic), list(tvalue), list(valid_factor_size), list(fpc)


def formuate_fitness_df(
        population_col_name: list, ic_fitness: list, tvalue_fitness: list, valid_factor_size: list,
        fpc: List[pd.Series],
        min_acceptable_factor_size: int,
        corr_thresh=0.9,
        save_path=None,
        logger=None):
    """将种群的fitness信息转化为DataFrame
    return_col: ['factor_name', 'ic', 'tvalue', 'valid_factor_size', 'fitness_score', 'selection_probability', 'cumulative_probability']
    要保证 fitness_df 的 index 和 self.population 一致
    """
    fitness_df = pd.DataFrame(
        columns=['factor_name', 'ic', 'tvalue', 'valid_factor_size'])
    fitness_df['factor_name'] = pd.Series(population_col_name)
    fitness_df['ic'] = pd.Series(ic_fitness)
    fitness_df['tvalue'] = pd.Series(tvalue_fitness)
    fitness_df['valid_factor_size'] = pd.Series(valid_factor_size)
    fitness_df.reset_index(drop=True, inplace=True)

    fitness_df['fitness_score'] = fitness_df['ic'].abs() + (np.tanh(fitness_df['tvalue'].abs()/3) * 3)
    # 有效因子个数小于阈值的，fitness_score置为0
    fitness_df['fitness_score'] = fitness_df['fitness_score'].mask(
        fitness_df['valid_factor_size'] < min_acceptable_factor_size, 0)

    # 筛选掉相关系数大的因子
    fpc = pd.concat(fpc, axis=1)
    corr_matrix = fpc.corr()
    sorted_feature_col = fitness_df.query('fitness_score > 0').sort_values(
        'fitness_score', ascending=False).factor_name.tolist()

    selected_features = CorrelationFilter.greedy_corr_filter(corr_matrix, corr_thresh, sorted_feature_col, True)
    
    if logger is not None:
        logger.info(
            f"clustering_corr_filter: {len(selected_features) / len(fitness_df) * 100:.2f}% of features are kept.")

    fitness_df['fitness_score'].mask(~fitness_df['factor_name'].isin(selected_features), 0, inplace=True)
    # 选择概率
    fitness_df['selection_probability'] = fitness_df['fitness_score'] / fitness_df['fitness_score'].sum()

    # 保存
    if save_path:
        fitness_df.to_csv(save_path, index=False, mode='a')

    return fitness_df


def test_pca(data):
    '''data should be demeaned'''
    try:
        C = np.dot(data.T, data) / (data.shape[0] - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(C)
        max_eigenvalue_index = np.argmax(eigenvalues)

        return eigenvectors[:, max_eigenvalue_index]
    except:

        pdb.set_trace()
