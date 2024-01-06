from Factors import ttest
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = '20'
sns.set()

logger = logging.getLogger('logger.EventStudy')


class EventStudy:

    @staticmethod
    def event_study(
            event_df: pd.DataFrame, ret_df: pd.DataFrame, ret_col: str, event_col: str, event_window=(-20, 20),
            group_col=None, date_col='TradingDay', secu_col='SecuCode', ret_interval=None, n_group=4,
            guess_win_rate=0.445, logger=logger):
        """Conduct event study

        Args:
            event_df (pd.DataFrame): _description_
            ret_df (pd.DataFrame): _description_
            ret_col (str): _description_
            event_col (str): _description_
            event_window (tuple, optional): _description_. Defaults to (-20, 20).
            group_col (_type_, optional): _description_. Defaults to None.
            date_col (str, optional): _description_. Defaults to 'TradingDay'.
            secu_col (str, optional): _description_. Defaults to 'SecuCode'.
            ret_interval (_type_, optional): _description_. Defaults to None.
            n_group (int, optional): _description_. Defaults to 4.
            guess_win_rate (float, optional): _description_. Defaults to 0.445.

        Returns:
            dict: keys = ['sum_df', 'fstats_df', 'infer_df']
                - sum_df: return of each day in event window, including mean, std, tvalue, win rate, p-value(win)
                - fstats_df: regression's F-statistic to group information as dummy variables
                - infer_df: recording each event's daily return around the occurrence of the event.
        """
        tmp_cols = [date_col, secu_col, event_col, group_col] if group_col is not None else [
            date_col, secu_col, event_col]
        df = ret_df[[date_col, secu_col, ret_col]].merge(
            event_df[tmp_cols], on=[date_col, secu_col], how='left')
        df[event_col] = df[event_col].fillna(0).astype(int)
        # 去掉未曾发生过事件的ticker, 减少计算量
        df = df[df.groupby(secu_col)[event_col].transform('sum') > 0]

        lag_df = EventStudy.create_lag(
            df, date_col, secu_col, ret_col, lag=event_window, inplace=False)
        # display(lag_df.head())
        lag_col = lag_df.columns[2:].to_list()
        if ret_interval is not None:
            logger.info(f'Create cumulative return for with given ret_interval: {ret_interval}')
            lag_col = []
            for val in ret_interval:
                cum_col = [
                    f"{ret_col}(t+{s})" if s >= 0 else f"{ret_col}(t{s})" for s in range(val[0], val[1])]
                new_col = '%s(%s,%s)' % (ret_col, val[0], val[1])
                lag_df[new_col] = lag_df[cum_col].mean(axis=1)
                # only valid if 2/3 value are not nan
                lag_df['valid'] = lag_df[cum_col].count(axis=1) >= (len(cum_col)*2/3)
                # replace value when condition is False
                lag_df[new_col] = lag_df[new_col].where(lag_df['valid'], np.nan)
                lag_df[new_col] = lag_df[new_col] * len(cum_col)  # 将日频变量转化为累计变量
                lag_col.append(new_col)

        tmp_df = df[df[event_col] == 1].copy()
        tmp_df = tmp_df.merge(lag_df, on=[date_col, secu_col], how='left')
        if group_col is not None:
            logger.info(f'Group by {group_col}')
            # group study
            if (pd.api.types.is_numeric_dtype(tmp_df[group_col])) and (tmp_df[group_col].nunique() > n_group*4):
                tmp_df[group_col] = pd.qcut(tmp_df[group_col], q=list(np.arange(
                    0, 1.01, 1/n_group)), labels=[f'G{i}' for i in range(1, n_group+1)])
            else:
                tmp_df[group_col] = tmp_df[group_col].astype('category')

            dummy_df = pd.get_dummies(tmp_df[group_col]).astype(int)
            dummy_col = dummy_df.columns.to_list()
            tmp_df = pd.concat([tmp_df, dummy_df], axis=1)

            f_stat = tmp_df[lag_col].apply(
                lambda x: sm.OLS(
                    x, sm.add_constant(tmp_df[dummy_col[1:]]),
                    missing='drop'
                ).fit().f_pvalue).rename('Fpval(mean)')
            mean = tmp_df.groupby(group_col)[
                lag_col].mean().unstack().rename('Mean(%)')
            clip_mean = tmp_df.groupby(group_col)[lag_col].agg(lambda x: x.clip(
                x.quantile(0.05), x.quantile(0.95)).mean()).unstack().rename('clip_mean(%)')
            t_stat = tmp_df.groupby(group_col)[lag_col].agg(lambda x: sm.OLS(endog=x.dropna(), exog=np.ones(
                x.dropna().shape), missing='none', hasconst=False).fit().tvalues[0]).unstack().rename('tvalue')
            t_stat2 = tmp_df.groupby(group_col)[lag_col].agg(
                lambda x: sm.OLS(
                    endog=x.clip(x.quantile(0.05), x.quantile(0.95)).dropna(),
                    exog=np.ones(x.dropna().shape), missing='none', hasconst=False
                ).fit().tvalues[0]).unstack().rename('tvalue(clip_mean)')
            f_stat2 = tmp_df[lag_col].apply(
                lambda x: sm.OLS(
                    x.clip(x.quantile(0.05), x.quantile(0.95)),
                    sm.add_constant(tmp_df[dummy_col[1:]]), missing='drop'
                ).fit().f_pvalue).rename('Fpval(clip_mean)')
            std = tmp_df.groupby(group_col)[
                lag_col].std().unstack().rename('Std(%)')
            q0 = tmp_df.groupby(group_col)[
                lag_col].min().unstack().rename('Min(%)')
            q25 = tmp_df.groupby(group_col)[lag_col].quantile(0.25).unstack().rename('q25(%)')
            q50 = tmp_df.groupby(group_col)[lag_col].quantile(0.5).unstack().rename('q50(%)')
            q75 = tmp_df.groupby(group_col)[lag_col].quantile(0.75).unstack().rename('q75(%)')
            q100 = tmp_df.groupby(group_col)[lag_col].max().unstack().rename('Max(%)')
            pos_ratio = tmp_df.groupby(group_col)[lag_col].apply(lambda x: (
                x > 0).sum()/x.notna().sum()).unstack().rename('WinRate(%)')
            count = tmp_df.groupby(group_col)[lag_col].count().unstack().rename('Count')
            prop_zval = tmp_df.groupby(group_col)[lag_col].agg(lambda x: proportions_ztest(
                (x.dropna() > 0).sum(), x.count(), guess_win_rate, 'two-sided')[1]).unstack().rename('p-val(win)')
            f_stat3 = tmp_df[lag_col].apply(
                lambda x: sm.OLS(
                    (x > 0).astype(int),
                    sm.add_constant(tmp_df[dummy_col[1:]]), missing='drop'
                ).fit().f_pvalue).rename('Fpval(win_rate)')
            sum_df = pd.concat([mean*100, clip_mean*100, std*100, t_stat, t_stat2, pos_ratio *
                               100, prop_zval, q0*100, q25*100, q50*100, q75*100, q100*100, count], axis=1)
            sum_df = sum_df.unstack()
            sum_df.index = lag_col
            sum_df.index.name = 'dist_to_event'
            sum_df2 = pd.concat([f_stat, f_stat2, f_stat3], axis=1)
            rst_dict = {'sum_df': sum_df, 'fstats_df': sum_df2, 'infer_df': tmp_df}
        else:
            mean = tmp_df[lag_col].mean().rename('Mean(%)')
            std = tmp_df[lag_col].std().rename('Std(%)')
            clip_mean = tmp_df[lag_col].apply(lambda x: x.clip(
                x.quantile(0.05), x.quantile(0.95)).mean()).rename('clip_mean(%)')
            t_stat = tmp_df[lag_col].apply(lambda x: sm.OLS(endog=x.dropna(), exog=np.ones(
                x.dropna().shape), missing='none', hasconst=False).fit().tvalues[0]).rename('tvalue')
            t_stat2 = tmp_df[lag_col].apply(
                lambda x: sm.OLS(
                    endog=x.clip(x.quantile(0.05), x.quantile(0.95)).dropna(),
                    exog=np.ones(x.dropna().shape), missing='none', hasconst=False
                ).fit().tvalues[0]).rename('tvalue(clip_mean)')

            q0 = tmp_df[lag_col].min().rename('Min(%) ')
            q25 = tmp_df[lag_col].quantile(0.25).rename('q25(%)')
            q50 = tmp_df[lag_col].quantile(0.5).rename('q50(%)')
            q75 = tmp_df[lag_col].quantile(0.75).rename('q75(%)')
            q100 = tmp_df[lag_col].max().rename('Max(%)')
            pos_ratio = tmp_df[lag_col].apply(lambda x: (x > 0).sum()/x.notna().sum()).rename('WinRate(%)')
            count = tmp_df[lag_col].count().rename('Count')
            prop_zval = tmp_df[lag_col].apply(lambda x: proportions_ztest(
                (x.dropna() > 0).sum(), x.count(), guess_win_rate, 'two-sided')[1]).rename('p-val(win)')
            sum_df = pd.concat([mean*100, clip_mean*100, std*100, t_stat, t_stat2, pos_ratio *
                               100, prop_zval, q0*100, q25*100, q50*100, q75*100, q100*100, count], axis=1)
            sum_df.index = lag_col
            sum_df.index.name = 'dist_to_event'
            rst_dict = {'sum_df': sum_df, 'infer_df': tmp_df}

        return rst_dict

    @staticmethod
    def event_study_fast(factor_df, ret_df, ret_col='excess_return', event_col='factor', ret_interval=(1, 10)):
        """
        分析信号后若干日个股平均表现，生成报表并画图。
        图中分别展示多头、空头和多空收益

        Args:
            factor_df (pd.DataFrame): factor, columns = ["SecuCode", "TradingDay", event_col]
            ret_df (pd.DataFrame): return, columns = ["SecuCode", "TradingDay", ret_col]
            ret_col (str, optional): _description_. Defaults to 'excess_return'.
            event_col (str, optional): _description_. Defaults to 'factor'.
            ret_interval (tuple, optional): _description_. Defaults to (1, 10).

        Returns:
            _type_: _description_
        """
        nd_ret_cols = []
        if isinstance(ret_interval, tuple):
            ret_interval = list(range(ret_interval[0], ret_interval[1] + 1))
        for itv in tqdm(ret_interval):
            col_name = f'cum({ret_col},{itv})'
            ret_df[col_name] = ret_df.groupby('SecuCode')[ret_col].rolling(itv).sum().droplevel(0)
            ret_df[col_name] = ret_df.groupby('SecuCode')[col_name].shift(itv-1)
            nd_ret_cols.append(col_name)

        tmp_df = factor_df.merge(
            ret_df[['TradingDay', 'SecuCode'] + nd_ret_cols], on=['TradingDay', 'SecuCode'], how='inner')
        # 总报表
        long_short_ret = tmp_df[nd_ret_cols].mean().rename(
            'total_long_short(%)') * 100
        short_ret = tmp_df[nd_ret_cols].mask(
            tmp_df[event_col] > 0).mean().rename('total_short(%)') * 100
        long_ret = tmp_df[nd_ret_cols].mask(
            tmp_df[event_col] < 0).mean().rename('total_long(%)') * 100
        tvalue = tmp_df[nd_ret_cols].apply(lambda x: ttest(x)).rename('total_tval')
        total_rst = pd.concat([long_short_ret, long_ret, short_ret, tvalue], axis=1).T

        # 年度报表
        tmp_df.set_index('TradingDay', inplace=True)
        annual_long_short = tmp_df[nd_ret_cols].resample('Y').mean() * 100
        annual_long_short.index = annual_long_short.index.year.map(
            lambda x: str(x) + '_long_short(%)')

        annual_long = tmp_df[nd_ret_cols].mask(
            tmp_df[event_col] < 0).resample('Y').mean() * 100
        annual_long.index = annual_long.index.year.map(
            lambda x: str(x) + '_long(%)')

        annual_short = tmp_df[nd_ret_cols].mask(
            tmp_df[event_col] > 0).resample('Y').mean() * 100
        annual_short.index = annual_short.index.year.map(
            lambda x: str(x) + '_short(%)')

        annual_rst = pd.concat([annual_long_short, annual_long, annual_short])

        # 由于return的计算方式是panelly的，所以要计算多头的比例
        xlab = [f"t+({i})" for i in ret_interval]
        long_prop = (tmp_df[event_col] > 0).sum() / tmp_df[event_col].count()
        plt.plot(xlab, long_short_ret.values, label='long_short')
        plt.plot(xlab, long_ret.values * long_prop, label='long')
        plt.plot(xlab, -short_ret.values * (1-long_prop), label='short')

        plt.fill_between(xlab, long_short_ret.values,
                         color='blue', alpha=0.2)
        plt.fill_between(xlab, long_ret.values *
                         long_prop, color='orange', alpha=0.2)
        plt.fill_between(xlab, -short_ret.values *
                         (1-long_prop), color='green', alpha=0.2)
        plt.xlabel('Days')
        plt.legend()
        plt.ylabel(f'{ret_col}(%)')
        plt.show()

        display(total_rst)

        return pd.concat([total_rst, annual_rst])

    @staticmethod
    def visualizeEventStudyResults(rst, title=None, event_window=(-20, 20)):
        '''
        对EventStudy 输出的结果进行可视化
        对于未分组的结果画出事件发生总收益以及25％分位数、75％分位数对于已分组的结果画出事件发生后的分组收益

        rst = event_study(event_df, ret_df, ret_col, event_col, 
            event_window=event_window, group_col=group_col, ret_interval=None)

        Be aware that the return_interval should be None!
        '''
        plot_df = rst['sum_df'][['Mean(%)', 'q25(%)', 'q75(%)']].copy()/100
        plot_df.rename(
            columns={'Mean(%)': 'mean', 'q25(%)': 'q25', 'q75(%)': 'q75'}, inplace=True)
        plot_df.index = np.arange(event_window[0], 0).tolist() + np.arange(1, event_window[1]+2).tolist()
        plot_df.loc[0, :] = 0
        plot_df.sort_index(inplace=True)
        ts1 = plot_df.loc[0:, 'mean']
        ts2 = plot_df.loc[event_window[0]:-1, 'mean']
        ts1 = ts1.cumsum()
        ts2 = (-ts2.sort_index(ascending=False)).cumsum()
        ts = pd.concat([ts1, ts2]).sort_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        ts.plot(ax=ax, label='平均收益')
        if plot_df.shape[1] == 3:
            # display(plot_df.head())
            ax.fill_between(ts.index, ts - plot_df['mean'] + plot_df['q25'],
                            ts - plot_df['mean'] + plot_df['q75'], alpha=0.3, color='orange')
        ax.set_title(title)

        return fig

    @staticmethod
    def create_lag(df, date_col, secu_col, lag_col, lag=10, inplace=True):
        """
        df: pd.DataFrame, columns = [date_col, entity_col, x_col], 
        lag: int, the number of lags to created for df
        return:
            df:pd.DataFrame, the dataframe with created lags
            name_list: list of str, the list of created lag column names 

        """
        df_list = []
        tmp_df = df.pivot(index=date_col, columns=secu_col, values=lag_col)
        if not isinstance(lag, tuple):
            lag = (1, lag)
        for la in range(lag[0], lag[1]+1):
            la_str = '+%s' % (la) if la >= 0 else '%s' % (la)
            xname = "%s(t%s)" % (lag_col, la_str)
            ts = tmp_df.shift(-la).unstack().rename(xname)
            df_list.append(ts)
        df_lag = pd.concat(df_list, axis=1).reset_index()
        if inplace:
            return df.merge(df_lag, on=[date_col, secu_col], how='left')
        else:
            return df_lag
