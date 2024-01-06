import matplotlib
matplotlib.use('agg')  # 'agg'是一种非交互式的后端

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from StockBackTest.LayerPlot import LayerPlot
import warnings
from IPython.display import display
import logging
import pandas as pd
from Factors import NumbaFuncs, ttest
from portfolio_optimizer.position import ControlUtils
warnings.filterwarnings('ignore')

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.rcParams['font.size'] = '20'  # 设置字体大小 = '16' # 设置字体大小
# sns.set()

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
FIG_SIZE = (8, 5)
try:
    import my_config as config
except:
    import config

    print('Failed to import module my_config.py, use default config file.')


def _load_returns(return_type='whole'):
    paths = {
        'whole': '20170101_20220630_WholeMarket_filted_T1_0930vwap_T2_0930vwap.csv',
        'hs300': '20170101_20220630_HS300_filted_T1_0930vwap_T2_0930vwap.csv',
        'zz500': '20170101_20220630_ZZ500_filted_T1_0930vwap_T2_0930vwap.csv',
        'zz1000': '20170101_20220630_ZZ1000_filted_T1_0930vwap_T2_0930vwap.csv'
    }
    fpath = os.path.join(os.path.dirname(__file__), '..', 'returns')
    return_df = pd.read_csv(os.path.join(
        fpath, paths.get(return_type)), index_col=0)
    return_df.index = return_df.index.values
    return_df = return_df.stack().reset_index()
    return_df.columns = ['TradingDay', 'SecuCode', 'return']
    return_df['TradingDay'] = return_df['TradingDay'].astype('datetime64[ns]')
    return_df['return'] = return_df['return'].replace(
        np.inf, 0).replace(-np.inf, 0)
    return_df['SecuCode'] = return_df['SecuCode'].map(
        lambda x: x.split('.')[0])

    return return_df.copy()


class BackTest:
    """
    date_col: TradingDay
    secu_col: SecuCode

    """

    def __init__(self, factor_df,
                 factor_name=None,
                 save_dir='../factor_logs/',
                 factor_lib_dir=None,
                 logger=logger,
                 plot=True,
                 folder_name=None,
                 ):
        '''
        factor_df: 
            SecuCode	TradingDay	`factor_name`
        19	000001	2017-07-28	1.712036
        20	000001	2017-07-31	1.860983
        21	000001	2017-08-01	2.043837
        默认取最后一列作为factor列
        '''

        self.factor_name = factor_df.columns[-1] if factor_name is None else factor_name
        self.factor_df = factor_df[[
            'TradingDay', 'SecuCode', self.factor_name]].dropna().copy()
        self.logger = logger
        self.factor_lib_dir = factor_lib_dir
        self.factor_df.columns = ['TradingDay', 'SecuCode', self.factor_name]
        if folder_name is None:
            self.save_dir = os.path.join(
                save_dir, f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{self.factor_name}")
        else:
            self.save_dir = os.path.join(save_dir, folder_name)
        self.plot = plot
        os.mkdir(self.save_dir)
        self.logger.info(f'Backtest Object created, factor name: {self.factor_name}; save dir: {self.save_dir}')

    def factor_analysis(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        factor_df = self.factor_df

        hist_fig = BTPlot.factor_hist(factor_df, self.factor_name, save_dir=self.save_dir)
        quantile_fig = BTPlot.factor_quantile(factor_df, self.factor_name, save_dir=self.save_dir)
        auto_corr_fig = BTPlot.factor_autocorr(factor_df, self.factor_name, save_dir=self.save_dir)
        stock_cnt_fig = BTPlot.factor_stock_count(factor_df, self.factor_name, save_dir=self.save_dir)
        if self.factor_lib_dir is not None:
            corr_fig = BTPlot.factor_corr(factor_df, self.factor_lib_dir, save_dir=self.save_dir)
        else:
            corr_fig = None
        # if self.plot:
        #     display(hist_fig)
        #     display(quantile_fig)
        #     display(auto_corr_fig)
        #     display(stock_cnt_fig)
        #     if corr_fig is not None:
        #         display(corr_fig)

        return [hist_fig, quantile_fig, stock_cnt_fig, auto_corr_fig, corr_fig]

    def ic_analysis(self, ret_df, rtype='excess_return', factor_freq='D', ret_shift=2):
        """计算IC

        Args:
            ret_df (_type_): _description_
            rtype (str, optional): _description_. Defaults to 'excess_return'.
            factor_freq (str, optional): _description_. Defaults to 'D'.
            ret_shift (int, optional): 收益向后shift天数,
                以raw_close_close 为例，T日对应的是T-1 到T日收益；
                对于T日计算得到的因子，需要在T+1日建仓，并赚取T+1到T+2 日的收益. Defaults to 2.
        """
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        ret_df = ret_df[['SecuCode', 'TradingDay', rtype]].copy()
        if ret_shift is not None:
            ret_df[rtype] = ret_df.groupby('SecuCode')[rtype].shift(-ret_shift)
        ret_df.dropna(inplace=True)

        tmp_df = self.factor_df.merge(ret_df, on=['SecuCode', 'TradingDay'], how='left').dropna()
        # 计算IC
        tmp_df['normalized_ret'] = tmp_df.groupby('TradingDay')[rtype].transform(
            lambda x: NumbaFuncs.normalize(x.values, False)).clip(-3, 3)
        tmp_df['normalized_factor'] = tmp_df.groupby('TradingDay')[self.factor_name].transform(
            lambda x: NumbaFuncs.normalize(x.values, False)).clip(-3, 3)
        tmp_df['ic'] = (tmp_df['normalized_factor'] * tmp_df['normalized_ret'])

        ic_ts = tmp_df.groupby('TradingDay')['ic'].mean()

        # 计算RankIC
        tmp_df['rank_ret'] = tmp_df.groupby('TradingDay')[rtype].rank(pct=True)
        tmp_df['rank_factor'] = tmp_df.groupby(
            'TradingDay')[self.factor_name].rank(pct=True)
        rankic_ts = tmp_df.groupby('TradingDay').apply(
            lambda x: np.mean((x['rank_factor'] - x['rank_factor'].mean()) / x['rank_factor'].std() *
                              (x['rank_ret'] - x['rank_ret'].mean()) / x['rank_ret'].std())
        )

        # 不同频率IC调整
        if factor_freq != 'D':
            ic_ts = ic_ts.resample(factor_freq).apply(
                lambda x: x.sum()/np.sqrt(x.count()))
            rankic_ts = rankic_ts.resample(factor_freq).apply(
                lambda x: x.sum()/np.sqrt(x.count()))

        # 全历史IC
        rst = {}
        rst['IC(%)'] = ic_ts.mean() * 100
        rst['t-value'] = ttest(ic_ts)
        ts_len = tmp_df['ic'].notna().sum()
        rst['IC+(%)'] = tmp_df['ic'].mask(tmp_df['normalized_factor'] < 0, np.nan).sum()/ts_len * 100
        rst['IC-(%)'] = tmp_df['ic'].mask(tmp_df['normalized_factor'] > 0, np.nan).sum()/ts_len * 100
        rst['RankIC(%)'] = rankic_ts.mean() * 100
        rst['ICIR'] = ic_ts.mean() / ic_ts.std()
        rst['Winning Rate(%)'] = (ic_ts > 0).mean() * 100
        rst['IC_Min(%)'] = ic_ts.min() * 100
        rst['IC_Max(%)'] = ic_ts.max() * 100
        wrapped_rst = pd.Series(rst).rename('total').to_frame().T
        # 记录因子到合并报表中
        logger.info(pd.Series(rst).rename(self.factor_name))

        # 逐年IC
        annual_rst = []
        annual_rst.append(
            (ic_ts.groupby(pd.Grouper(freq='Y')).mean() * 100).rename('IC(%)'))
        annual_rst.append(ic_ts.groupby(pd.Grouper(freq='Y')).apply(
            lambda x: ttest(x)).rename('t-value'))
        annual_rst.append(
            tmp_df.set_index('TradingDay').groupby(pd.Grouper(freq='Y')).apply(lambda x: x['ic'].mask(
                x['normalized_factor'] < 0, np.nan).sum() / x['ic'].notna().sum()).rename('IC+(%)') * 100
        )
        annual_rst.append(
            tmp_df.set_index('TradingDay').groupby(pd.Grouper(freq='Y')).apply(lambda x: x['ic'].mask(
                x['normalized_factor'] > 0, np.nan).sum() / x['ic'].notna().sum()).rename('IC-(%)') * 100
        )
        annual_rst.append(
            (rankic_ts.groupby(pd.Grouper(freq='Y')).mean() * 100).rename('RankIC(%)')
        )
        annual_rst.append(
            ic_ts.groupby(pd.Grouper(freq='Y')).apply(
                lambda x: x.mean() / x.std()).rename('ICIR')
        )
        annual_rst.append(ic_ts.groupby(pd.Grouper(freq='Y')).apply(
            lambda x: (x > 0).mean()).rename('Winning Rate(%)') * 100)
        annual_rst.append(ic_ts.groupby(pd.Grouper(freq='Y')
                                        ).min().rename('IC_Min(%)') * 100)
        annual_rst.append(ic_ts.groupby(pd.Grouper(freq='Y')
                                        ).max().rename('IC_Max(%)') * 100)
        annual_rst = pd.concat(annual_rst, axis=1)
        annual_rst.index = annual_rst.index.year

        #### 生成报表 ####
        # IC报表, total + annual
        ic_rst = pd.concat([wrapped_rst, annual_rst])
        ic_rst.to_csv(os.path.join(self.save_dir, 'ic_rst.csv'))
        layered_ic_plot = BTPlot.group_box_plot(tmp_df, 'normalized_factor',
                                                rtype, 'normalized_ret',
                                                save_dir=self.save_dir)
        ic_decay_plot = BTPlot.ic_decay_plot(tmp_df, factor_freq, save_dir=self.save_dir)
        ic_cumsum_plot = BTPlot.ic_cumsum_plot(ic_ts, save_dir=self.save_dir)
        if self.plot:
            display(ic_rst.round(3))
            # display(layered_ic_plot)
            # display(ic_decay_plot)
            # display(ic_cumsum_plot)
        return {'ic_rst': ic_rst,
                'ic_ts': ic_ts,
                'figs': [layered_ic_plot, ic_decay_plot, ic_cumsum_plot]}

    def return_analysis(self, ret_df, rtype='excess_return',
                        normalize=True,rank=False, ret_shift=2,fee=0.0005,
                        slippage=0.001,topn_stocks=None,single_stock_max_weight=0.05,
                        inverse=False):
        """
        topn_stocks: 每日只多空分别只持仓topn_stocks只股票, 取值为None时不限制
        TODO: 滑点、涨跌停、最大持仓比例
        normalize: 对因子去均值、标准差
        """
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        ret_df = ret_df[['SecuCode', 'TradingDay', rtype]].copy()
        if ret_shift is not None:
            ret_df[rtype] = ret_df.groupby('SecuCode')[rtype].shift(-ret_shift)
        ret_df.dropna(inplace=True)

        factor_df = self.factor_df[['SecuCode','TradingDay',self.factor_name]].copy()
        
        if inverse:
            factor_df[self.factor_name] = - factor_df[self.factor_name]
        if rank:
            factor_df[self.factor_name] = factor_df.groupby('TradingDay')[self.factor_name].transform(
                lambda x: x.rank(pct=True)) - 0.5
            normalize = False # 不再对因子进行normalize
            
        if normalize:
            factor_df[self.factor_name] = factor_df.groupby('TradingDay')[self.factor_name].transform(
                lambda x: NumbaFuncs.normalize(x.values, False)).clip(-3, 3)
        tmp_df = ret_df.merge(
            factor_df, on=['SecuCode', 'TradingDay'], how='left') # 此处不可dropna 否则会对仓位计算造成影响
        
        tmp_df['short_factor'] = -tmp_df[self.factor_name].mask(tmp_df[self.factor_name] > 0, np.nan) # 取反, 保证short_factor为正
        tmp_df['long_factor'] = tmp_df[self.factor_name].mask(tmp_df[self.factor_name] < 0, np.nan)
        if topn_stocks is not None:
            tmp_df['short_rank'] = tmp_df.groupby('TradingDay')['short_factor'].rank(ascending=False)
            tmp_df['long_rank'] = tmp_df.groupby('TradingDay')['long_factor'].rank(ascending=False)
            tmp_df['short_factor'] = tmp_df['short_factor'].mask(tmp_df['short_rank'] > topn_stocks, np.nan)
            tmp_df['long_factor'] = tmp_df['long_factor'].mask(tmp_df['long_rank'] > topn_stocks, np.nan)
            
        tmp_df['short_position'] = tmp_df.groupby('TradingDay')['short_factor'].transform(lambda x: x/x.sum()).fillna(0)
        tmp_df['long_position'] = tmp_df.groupby('TradingDay')['long_factor'].transform(lambda x: x/x.sum()).fillna(0)
        if single_stock_max_weight is not None:
            # 限制最大仓位
            tmp_df['short_position'] = tmp_df.groupby('TradingDay')['short_position'].transform(lambda x: ControlUtils.limit_weights(x.values,threshold=single_stock_max_weight))
            tmp_df['long_position'] = tmp_df.groupby('TradingDay')['long_position'].transform(lambda x: ControlUtils.limit_weights(x.values,threshold=single_stock_max_weight))
        tmp_df['short_position'] = - tmp_df['short_position']
        
        tmp_df['long_ret'] = tmp_df['long_position'] * tmp_df[rtype]
        tmp_df['short_ret'] = tmp_df['short_position'] * tmp_df[rtype]
        ret = tmp_df.groupby('TradingDay')[['long_ret', 'short_ret']].sum()
        ret['long_short_ret'] = ret['long_ret'] + ret['short_ret']
        
        # 算turnover
        tmp_df['total_position'] = (tmp_df['short_position'].fillna(0) + tmp_df['long_position'].fillna(0))
        tmp_df['delta_position'] = tmp_df.groupby('SecuCode')['total_position'].diff().abs()

        ret['turnover'] = tmp_df.groupby('TradingDay')['delta_position'].sum()
        ret['long_short_ret_fee'] = ret['long_short_ret'] - ret['turnover'] * fee
        rst = {}
        # 总量统计量, 在因子间不可比
        rst['WinningRate(%)'] = (ret['long_short_ret'] > 0).mean() * 100
        rst['tot_long_short_ret(%)'] = ((1 + ret['long_short_ret']).prod() - 1) * 100
        rst['tot_long_ret(%)'] = ((1 + ret['long_ret']).prod() - 1) * 100
        rst['tot_short_ret(%)'] = ((1 + ret['short_ret']).prod() - 1) * 100
        rst['tot_long_short_ret_fee(%)'] = ((1 + ret['long_short_ret_fee']).prod() - 1) * 100
        
        # 年化统计量
        ndays = len(ret)
        rst['long_short_ret(%)'] = (np.power(((1 + ret['long_short_ret']).prod()), 242/ndays) - 1) * 100
        rst['long_short_ret_fee(%)'] = (np.power(((1 + ret['long_short_ret_fee']).prod()), 242/ndays) - 1) * 100
        rst['IR'] = rst['long_short_ret(%)'] / \
            (ret['long_short_ret'].std() * np.sqrt(ndays)*100)
        rst['long_ret(%)'] = (np.power(((1 + ret['long_ret']).prod()), 242/ndays) - 1) * 100
        rst['short_ret(%)'] = (
            np.power(((1 + ret['short_ret']).prod()), 242/ndays) - 1) * 100
        
        rst['WinningRate(%)'] = (ret['long_short_ret'] > 0).mean() * 100
        rst['turnover(%)'] = ret['turnover'].mean() * 100 / 2
        rst['Sharpe'] = (ret['long_short_ret'].mean() / ret['long_short_ret'].std()) * np.sqrt(242)
        
        rst_se = pd.Series(rst).rename('totoal').to_frame().T
        # 记录return log
        self.logger.info(pd.Series(rst).rename(self.factor_name))

        # 逐年return
        annual_rst = []
        annual_rst.append((ret.groupby(pd.Grouper(freq='Y'))['long_ret'].apply(
            lambda x: NumbaFuncs.annual_return(x.values) * 100)).rename('long_ret(%)'))
        annual_rst.append((ret.groupby(pd.Grouper(freq='Y'))['short_ret'].apply(
            lambda x: NumbaFuncs.annual_return(x.values) * 100).rename('short_ret(%)')))
        annual_rst.append((ret.groupby(pd.Grouper(freq='Y'))['long_short_ret'].apply(
            lambda x: NumbaFuncs.annual_return(x.values) * 100).rename('long_short_ret(%)')))
        annual_rst.append((ret.groupby(pd.Grouper(freq='Y'))['long_short_ret_fee'].apply(
            lambda x: NumbaFuncs.annual_return(x.values) * 100).rename('long_short_ret_fee(%)')))
        annual_rst.append(ret.groupby(pd.Grouper(freq='Y'))['turnover'].mean().rename('turnover(%)') * 100/2)
        annual_rst.append((ret.groupby(pd.Grouper(freq='Y'))['long_short_ret'].apply(
            lambda x: (x > 0).mean()) * 100).rename('WinningRate(%)'))

        annual_rst.append(ret.groupby(pd.Grouper(freq='Y'))['long_short_ret'].apply(
            lambda x: x.mean()/x.std() * np.sqrt(242)).rename('Sharpe'))
        annual_rst.append((annual_rst[2] / (ret.groupby(pd.Grouper(freq='Y'))[
                          'long_short_ret'].std() * np.sqrt(242) * 100)).rename('IR'))
        annual_rst = pd.concat(annual_rst, axis=1)
        annual_rst.index = annual_rst.index.year
        return_rst = pd.concat([rst_se, annual_rst])[['long_short_ret(%)', 'long_ret(%)', 'short_ret(%)', 'long_short_ret_fee(%)',
                                                      'WinningRate(%)','turnover(%)', 'Sharpe','IR', 'tot_long_short_ret(%)', 'tot_long_ret(%)', 'tot_short_ret(%)']]
        
        ret_long_short_plot = BTPlot.ret_long_short_plot(ret, save_dir=self.save_dir)

        #     # 画 cum_return 图
        # if universe is not None:
        #     fig = BackTest.group_layer_plot(self.factor_df, universe=universe)
        #     if self.save_rst:
        #         fig.savefig(os.path.join(self.save_dir, 'CumReturn.png'))
        #     # display(fig)
        #     if self.remove_rst:
        #         shutil.rmtree(self.save_dir)
        #         # os.rmdir(self.save_dir)
        if self.plot:
            display(return_rst.round(3))
            # display(ret_long_short_plot)
        
        if self.save_dir is not None:
            return_rst.to_csv(os.path.join(self.save_dir, 'ret_rst.csv'))
            
        return {'return_rst': return_rst, 'daily_df': ret, 'figs': [ret_long_short_plot]}

    def wrapped_analysis(self, ret_df, rtype='excess_return', ret_shift=2, factor_freq='D',normalize=True,fee=0.0005):
        factor_rst = self.factor_analysis()
        ic_rst = self.ic_analysis(ret_df, rtype=rtype,
                         ret_shift=ret_shift,factor_freq=factor_freq)
        
        if ic_rst['ic_rst'].loc['total','IC(%)'] < 0:
            self.logger.info(f"Factor: {self.factor_name} IC < 0, treat as inverse factor.")
            inverse = True
        else:
            inverse = False
        ret_rst = self.return_analysis(ret_df, rtype=rtype, ret_shift=ret_shift,normalize=normalize,inverse=inverse,fee=fee,)
        
        return {'factor_rst': factor_rst, 'ic_rst': ic_rst, 'ret_rst': ret_rst} 


    @staticmethod
    def batch_ic_analysis(factor_df, ret_df=None, 
                          rtype='excess_return', plot=True,
                          ic_thresh=0, ret_shift=None,
                          force_calc_corr=False):
        """
        对于含有多个因子的factor_df 进行快速分析
        注意return 需要先进行 shift; ret_df 可以缺省
        """
        factor_df = factor_df.copy()
        if ret_df is None:

            if rtype not in factor_df.columns:
                raise ValueError(
                    'ret_df is None and rtype not in factor_df.columns')
            else:
                factor_col = set(factor_df.columns.tolist()) - \
                    set(['TradingDay', 'SecuCode', rtype])
            if ret_shift is not None:
                raise Warning(
                    'ret_df is None and ret_shift is not None, ret_shift will be ignored')
        else:
            ret_df.dropna(subset=['SecuCode', 'TradingDay', rtype], inplace=True)
            if ret_shift is not None:
                ret_df = ret_df.copy()
                ret_df[rtype] = ret_df.groupby('SecuCode')[rtype].shift(-ret_shift)

            factor_col = set(factor_df.columns.tolist()) - set(['TradingDay', 'SecuCode'])
            factor_df = factor_df.merge(ret_df[['SecuCode', 'TradingDay', rtype]],
                                        on=['SecuCode', 'TradingDay'], how='left')

        factor_col = list(factor_col)
        factor_df.dropna(subset=factor_col, how='all', inplace=True)

        def analysis_one_factor(col):
            tmp_df = factor_df[['TradingDay', col,rtype]].dropna().copy()

            # crossectional normalized return 对每个因子需要单独计算
            normalized_ret_col = f'__normalized{rtype}'
            tmp_df[normalized_ret_col] = tmp_df.groupby('TradingDay')[rtype].transform(
                lambda x: NumbaFuncs.normalize(x.values, False)).clip(-3, 3)

            normalized_rank_ret_col = f'__normalized_rank{rtype}'
            tmp_df[normalized_rank_ret_col] = tmp_df.groupby('TradingDay')[rtype].rank(pct=True)
            tmp_df[normalized_rank_ret_col] = tmp_df.groupby('TradingDay')[normalized_rank_ret_col].transform(
                lambda x: NumbaFuncs.normalize(x.values, False))

            # 计算IC
            rst = {}

            tmp_df['ic'] = (tmp_df[col] * tmp_df[normalized_ret_col])
            ic_ts = tmp_df.groupby('TradingDay')['ic'].mean()
            rst['IC(%)'] = ic_ts.mean() * 100
            if np.abs(rst['IC(%)']) < ic_thresh:
                # 如果IC太小，直接返回nan
                rst['t-value'] = np.nan
                rst['IC+(%)'] = np.nan
                rst['IC-(%)'] = np.nan
                rst['RankIC(%)'] = np.nan
                rst['ICIR'] = np.nan
                rst['Winning Rate(%)'] = np.nan
                rst['IC_Min(%)'] = np.nan
                rst['IC_Max(%)'] = np.nan
                rst['AveragedDailyFactorCount'] = np.nan

                return pd.Series(rst).rename(col)

            tmp_df['rank_factor'] = tmp_df.groupby(
                'TradingDay')[col].rank(pct=True)
            tmp_df['rank_factor'] = tmp_df.groupby('TradingDay')['rank_factor'].transform(
                lambda x: NumbaFuncs.normalize(x.values, False))
            tmp_df['rank_ic'] = tmp_df['rank_factor'] * tmp_df[normalized_rank_ret_col]
            rankic_ts = tmp_df.groupby('TradingDay')['rank_ic'].mean()

            rst['t-value'] = ttest(ic_ts)
            ts_len = tmp_df['ic'].notna().sum()
            rst['IC+(%)'] = tmp_df['ic'].mask(tmp_df[col]
                                              < 0, np.nan).sum()/ts_len * 100
            rst['IC-(%)'] = tmp_df['ic'].mask(tmp_df[col]
                                              > 0, np.nan).sum()/ts_len * 100
            rst['RankIC(%)'] = rankic_ts.mean() * 100
            rst['ICIR'] = ic_ts.mean() / ic_ts.std()
            rst['Winning Rate(%)'] = (ic_ts > 0).mean() * 100
            rst['IC_Min(%)'] = ic_ts.min() * 100
            rst['IC_Max(%)'] = ic_ts.max() * 100
            rst['AveragedDailyFactorCount'] = tmp_df.groupby('TradingDay')[col].count().mean()
            
            return pd.Series(rst).rename(col)

        result_list = []
        print('Calculating IC')
        error_columns = []
        for col in tqdm(factor_col):
            try:
                factor_df[col] = factor_df.groupby('TradingDay')[col].transform(
                    lambda x: NumbaFuncs.normalize(x.values, clip=False))
                result_list.append(analysis_one_factor(col))
            except Exception as e:
                # print(f'Error in processing {col}; Error: {e}')
                error_columns.append(col)
        print(f"Finish calculating IC, {len(error_columns)} columns failed")
        ic_rst = pd.concat(result_list, axis=1).T.sort_values(
            'IC(%)', ascending=False,key=abs)
        ic_rst = ic_rst[ic_rst['IC(%)'].map(lambda x: np.abs(x) > ic_thresh)]
        factor_col = list(ic_rst.index.values)
        if len(factor_col) <=30 or force_calc_corr:
            corr_rst = factor_df[factor_col].corr()
            corr_plot =BTPlot.plot_corr(corr_rst, save_dir=None)
        else:
            corr_rst = None
            
        if plot:
            display(ic_rst.round(3))
            
                
        return {'ic_rst': ic_rst,
                'corr_rst': corr_rst,
                'corr_plot': corr_plot,
                'error_columns': error_columns}


class BTPlot:
    """
    BTPlot: BackTest Plot
    factor_df: 因子数据, 包含SecuCode, TradingDay, factor;
    ic_calc_df: 计算IC的数据, 包含SecuCode, TradingDay, normalized_ret, normalized_factor;

    """
    @staticmethod
    def factor_hist(factor_df, factor_name, save_dir=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZE)

        ax.hist(factor_df[factor_name], bins=20)
        ax.set_title(f'Factor Histogram: {factor_name}')

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f'FactorHistogram.png'))

        return fig

    @staticmethod
    def factor_quantile(factor_df, factor_name, save_dir=None, ax=None):
        daily_quantile_df = factor_df.groupby('TradingDay')[factor_name].agg([
            ('q80', lambda x: x.quantile(0.8)),
            ('q60', lambda x: x.quantile(0.6)),
            ('q50', lambda x: x.quantile(0.5)),
            ('mean', lambda x: x.mean()),
            ('q40', lambda x: x.quantile(0.4)),
            ('q20', lambda x: x.quantile(0.2)),
        ])
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(daily_quantile_df)
        ax.set_title(f"Factor's Value Over Time: {factor_name}")

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f'QuantilePlot.png'))
        return fig

    @staticmethod
    def factor_stock_count(factor_df, factor_name, date_col='TradingDay', save_dir=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
        factor_df_exzero = factor_df[factor_df[factor_name] != 0]
        factor_df.groupby(date_col)[factor_name].count().rename('stock count').plot(ax=ax)
        factor_df_exzero.groupby(date_col)[factor_name].count().rename('stock count(exclude zero)').plot(ax=ax)
        ax.set_title(f'Factor Stock Count: {factor_name}')
        ax.legend()

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f'FactorStockCount.png'))
        return fig

    @staticmethod
    def factor_autocorr(factor_df, factor_name, lags=(1, 10), date_col='TradingDay', secu_col='SecuCode', save_dir=None):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        lags = (1, 10)
        bar_plot_data = {}
        for t in range(lags[0], lags[1]+1):
            bar_plot_data[f'T+{t}'] = factor_df.set_index(date_col).groupby(
                secu_col)[factor_name].apply(lambda x: NumbaFuncs.auto_corr(x.values, t)).mean()

        pd.Series(bar_plot_data).plot(kind='bar', color='b', ax=ax)
        ax.set_title(f'Factor Autocorrelation: {factor_name}')
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f'FactorAutoCorr.png'))
        return fig

    @staticmethod
    def factor_corr_plot(factor_df, factor_lib_dir, date_col='TradingDay', secu_col='SecuCode', save_dir=None):
        all_factors = os.listdir(factor_lib_dir)
        all_factors = [i for i in all_factors if i.endswith('.pkl')]
        df = factor_df
        for file in all_factors:
            df = df.merge(pd.read_pickle(os.path.join(factor_lib_dir, file)), on=[
                          secu_col, date_col], how='left')
        rst = df.corr()
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        sns.heatmap(rst.round(2), annot=True, ax=ax)
        if save_dir is not None:
            fig.savefig(save_dir=os.path.join(save_dir, 'FactorCorr.png'))

        return fig

    @staticmethod
    def group_box_plot(ic_calc_df, factor_name, return_col='raw_close_close', norm_ret_col='normalized_ret', save_dir=None):
        ic_calc_df['factor_group'] = np.nan
        ic_calc_df['factor_group'] = ic_calc_df['factor_group'].fillna(pd.qcut(ic_calc_df[factor_name].mask(
            ic_calc_df[factor_name] > 0, np.nan), 10, labels=list(range(1, 11))).astype(str)
        )
        ic_calc_df['factor_group'] = ic_calc_df['factor_group'].replace('nan', np.nan)
        ic_calc_df['factor_group'] = ic_calc_df['factor_group'].fillna(
            pd.qcut(ic_calc_df[factor_name].mask(
                ic_calc_df[factor_name] <= 0, np.nan), 10, labels=list(range(11, 21))).astype(str)
        )
        ic_calc_df['factor_group'] = ic_calc_df['factor_group'].replace('nan', np.nan)
        # 箱线图
        fig, axs = plt.subplots(1, 2, figsize=(FIG_SIZE[0]*2 + 1, FIG_SIZE[1]))
        for ax, ret in zip(axs, [return_col, norm_ret_col]):
            plot_data = {}
            for group, data in ic_calc_df.groupby('factor_group')[ret]:
                data.index = list(range(len(data)))
                plot_data[int(float(group))] = data
            plot_df = pd.DataFrame(plot_data)
            plot_df = plot_df[plot_df.columns.sort_values()]

            sns.boxplot(data=plot_df.clip(plot_df.quantile(0.01), plot_df.quantile(0.99), axis=1),
                        fliersize=0, showmeans=True, notch=True,
                        boxprops=dict(facecolor='none', edgecolor='grey'), width=0.8, ax=ax)
            ax.plot(list(range(-1, 21)), [0]*22, color='black', label='zero', alpha=0.8)
            ax.plot(list(range(20)), plot_df.mean().values, color='red', label='mean')
            ax.set_xlim(-1, 20)
            ax.set_ylabel(ret)
            ax.set_xlabel('Factor Group')
            ax.set_title(f'Layered Return: {ret}')
            ax.grid(True)

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, 'LayeredReturn.png'))
        return fig

    @staticmethod
    def ic_decay_plot(ic_calc_df, factor_freq='D', lags=(-5, 10), save_dir=None):

        plot_df = ic_calc_df[['TradingDay', 'SecuCode', 'normalized_ret', 'normalized_factor']].copy()
        rst = {}
        for lag in range(lags[0], lags[1]+1+1):
            col_name = f"T+{lag}"
            plot_df[col_name] = plot_df.groupby('SecuCode')['normalized_ret'].shift(-lag)
            plot_df[col_name] = (plot_df[col_name] * plot_df['normalized_factor'])
            ic_ts = plot_df.groupby('TradingDay')[col_name].mean()
            rst[col_name] = ic_ts.mean()*100 if factor_freq == 'D' else ic_ts.resample(
                factor_freq).apply(lambda x: x.sum()/np.sqrt(x.count())).mean() * 100
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        pd.Series(rst).plot(kind='bar', color='b', ax=ax)

        ax.set_ylabel('IC(%)')
        ax.set_title('IC Decay Plot')
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, 'DecayIC.png'))
        return fig

    @staticmethod
    def ic_cumsum_plot(ic_ts, save_dir=None):
        ic_monthly = ic_ts.resample('M').mean()
        ic_cumsum = ic_ts.cumsum()
        fig, ax1 = plt.subplots(figsize=FIG_SIZE)
        ax2 = ax1.twinx()
        ax1.plot(ic_cumsum, label='Cumulative IC')
        ax1.set_ylabel('Cumulative IC')
        
        ax2.bar(ic_monthly.index, ic_monthly.values * 100, width=16,alpha=0.3, color='purple',label='Monthly Averaged IC(%) (Right)')
        ax2.set_ylabel('Monthly Averaged IC(%)')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax2.grid(False)
        ax2.set_ylim(-10,10)
        ax1.set_title('Cumulative IC Plot')

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, 'CumulativeIC.png'))
        return fig

    @staticmethod
    def ret_long_short_plot(ret,save_dir=None):
        fig,ax1 = plt.subplots(figsize=FIG_SIZE)
        ax2 = ax1.twinx()
        monthly_long_short = ret['long_short_ret'].resample('M').sum()
        ax1.plot(ret[['long_short_ret','long_short_ret_fee', 'long_ret', 'short_ret']].cumsum(), label=['long_short_ret','long_short_ret_fee', 'long_ret', 'short_ret'])
        ax2.bar(monthly_long_short.index, monthly_long_short.values, width=16, alpha=0.3, color='purple',label='Monthly Long Short (Right)')
        
        ax1.legend()
        ax2.legend()
        ax1.set_ylabel('Cumulative Return')
        ax2.set_ylabel('Monthly Long Short Return')
        ax1.set_title('Long Short Return Plot')
        ax2.grid(False)
        ax2.set_ylim(-0.2,0.2)
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, 'LongShortReturn.png'))
        
        return fig
    
    @staticmethod
    def plot_corr(corr_df,save_dir=None):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        annot = corr_df.shape[0] < 10 # 因子数量小于10时才显示annot
        sns.heatmap(corr_df, 
                    vmin=-1, vmax=1, center=0,
                    cmap=sns.diverging_palette(10, 220, sep=80, n=8),
                    annot=annot, fmt=".2f",
                    ax=ax)
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, 'FactorCorr.png'))
        return fig
    
    @staticmethod
    def group_layer_plot(factor_df, groups=20, factor_name='factor',universe='all'):

        def gen_colors(groups):
            green_count = (int)(groups/2)
            red_count = groups - green_count

            bound = 150
            green_values = [int(i*bound/green_count)
                            for i in range(green_count)]
            red_values = [int(i*bound/red_count) for i in range(red_count)]

            # RGB值越大越亮越浅
            green_colors = ["#%02x%02x%02x" %
                            (0, 250 - bound + int(g), 0) for g in green_values]
            red_colors = ["#%02x%02x%02x" %
                          (250 - int(r), 0, 0) for r in red_values]
            market = ["#%02x%02x%02x" % (0, 0, 250)]
            colors = green_colors + red_colors + market
            colors.append(market)
            return colors[0:-1]
        if universe == 'all':
            fig = plt.figure(figsize=(16, 6 * 4))
            ax1 = plt.subplot(4, 1, 1)
            colors = gen_colors(groups)
            concat_df = factor_df.merge(
                _load_returns(), on=['SecuCode', 'TradingDay'], how='left').dropna()

            pct_df = LayerPlot.get_layer_returns(
                concat_df.set_index('TradingDay'), groups=groups)
            cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
            cumret_df.plot(color=colors, ax=ax1)
            figname = 'cumRet:{}'.format(
                str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
            ax1.set_title('{} {}'.format('whole market', figname))


            ax2 = plt.subplot(4, 1, 2)
            colors = gen_colors(groups)
            concat_df = factor_df.merge(_load_returns('hs300'), on=[
                                        'SecuCode', 'TradingDay'], how='left').dropna()
            pct_df = LayerPlot.get_layer_returns(
                concat_df.set_index('TradingDay'), groups=groups)
            cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
            cumret_df.plot(color=colors, ax=ax2)
            figname = 'cumRet:{}'.format(
                str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
            ax2.set_title('{} {}'.format('HS300', figname))

            ax3 = plt.subplot(4, 1, 3)
            colors = gen_colors(groups)
            concat_df = factor_df.merge(_load_returns('zz500'), on=[
                                        'SecuCode', 'TradingDay'], how='left').dropna()
            pct_df = LayerPlot.get_layer_returns(
                concat_df.set_index('TradingDay'), groups=groups)
            cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
            cumret_df.plot(color=colors, ax=ax3)
            figname = 'cumRet:{}'.format(
                str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
            ax3.set_title('{} {}'.format('ZZ500', figname))

            ax4 = plt.subplot(4, 1, 4)
            colors = gen_colors(groups)
            concat_df = factor_df.merge(_load_returns('zz1000'), on=[
                                        'SecuCode', 'TradingDay'], how='left').dropna()
            pct_df = LayerPlot.get_layer_returns(
                concat_df.set_index('TradingDay'), groups=groups)
            cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
            cumret_df.plot(color=colors, ax=ax4)
            figname = 'cumRet:{}'.format(
                str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
            ax4.set_title('{} {}'.format('ZZ1000', figname))

            return fig
        elif universe == 'whole market':
            fig, ax1 = plt.subplots(figsize=(16, 6))
            colors = gen_colors(groups)
            concat_df = factor_df.merge(
                _load_returns(), on=['SecuCode', 'TradingDay'], how='left').dropna()

            pct_df = LayerPlot.get_layer_returns(
                concat_df.set_index('TradingDay'), groups=groups)
            cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
            cumret_df.plot(color=colors, ax=ax1)
            figname = 'cumRet:{}'.format(
                str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
            ax1.set_title('{} {}'.format('whole market', figname))
        elif universe == 'hs300':
            fig, ax2 = plt.subplots(figsize=(16, 6))
            colors = gen_colors(groups)
            concat_df = factor_df.merge(_load_returns('hs300'), on=[
                                        'SecuCode', 'TradingDay'], how='left').dropna()
            pct_df = LayerPlot.get_layer_returns(
                concat_df.set_index('TradingDay'), groups=groups)
            cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
            cumret_df.plot(color=colors, ax=ax2)
            figname = 'cumRet:{}'.format(
                str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
            ax2.set_title('{} {}'.format('HS300', figname))
        elif universe == 'zz500':
            fig, ax3 = plt.subplots(figsize=(16, 6))
            colors = gen_colors(groups)
            concat_df = factor_df.merge(_load_returns('zz500'), on=[
                                        'SecuCode', 'TradingDay'], how='left').dropna()
            pct_df = LayerPlot.get_layer_returns(
                concat_df.set_index('TradingDay'), groups=groups)
            cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
            cumret_df.plot(color=colors, ax=ax3)
            figname = 'cumRet:{}'.format(
                str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
            ax3.set_title('{} {}'.format('ZZ500', figname))
        elif universe == 'zz1000':
            fig, ax4 = plt.subplots(figsize=(16, 6))
            colors = gen_colors(groups)
            concat_df = factor_df.merge(_load_returns('zz1000'), on=[
                                        'SecuCode', 'TradingDay'], how='left').dropna()
            pct_df = LayerPlot.get_layer_returns(
                concat_df.set_index('TradingDay'), groups=groups)
            cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
            cumret_df.plot(color=colors, ax=ax4)
            figname = 'cumRet:{}'.format(
                str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
            ax4.set_title('{} {}'.format('ZZ1000', figname))
        else:
            raise ValueError(
                'universe must be in [all, whole market, hs300, zz500, zz1000]')
            
    

        
