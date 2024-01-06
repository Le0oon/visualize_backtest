# Author: Qize Liu
# Description: Operators for AlphaTree

import pandas as pd
import numpy as np
import numba
from Factors import NumbaFuncs


class SingleAttrTsOperator:
    """
    Calculate corresponding col value then return col name.
    input: df, fcol, n
    return: col_name
    """

    @staticmethod
    def ts_mean(df, fcol, n):
        """计算n日滚动均值
        """
        col_name = 'ts_mean(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(n).mean().droplevel(0)

        return col_name

    @staticmethod
    def ts_std(df, fcol, n):
        """计算n日滚动标准差
        """
        col_name = 'ts_std(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(n).std().droplevel(0)

        return col_name

    @staticmethod
    def ts_max(df, fcol, n):
        """计算n日滚动最大值
        """
        col_name = 'ts_max(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(n).max().droplevel(0)

        return col_name

    @staticmethod
    def ts_min(df, fcol, n):
        """计算n日滚动最小值
        """
        col_name = 'ts_min(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(n).min().droplevel(0)

        return col_name

    @staticmethod
    def ts_median(df, fcol, n):
        """计算n日滚动中位数
        """
        col_name = 'ts_median(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(n).median().droplevel(0)

        return col_name

    @staticmethod
    def ts_skew(df, fcol, n):
        """计算n日滚动偏度
        """
        col_name = 'ts_skew(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(n).skew().droplevel(0)

        return col_name

    @staticmethod
    def ts_kurt(df, fcol, n):
        """计算n日滚动峰度
        """
        col_name = 'ts_kurt(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(n).kurt().droplevel(0)

        return col_name

    @staticmethod
    def ts_autocorr(df, fcol, n):
        """计算过去n日的一阶自相关系数
        """
        if n < 10:
            n += 10

        col_name = 'ts_autocorr(%s,%s)' % (fcol, n)
        if col_name not in df.columns:
            df[col_name] = df.groupby(
                'SecuCode')[fcol].rolling(n).apply(lambda x: NumbaFuncs.auto_corr(x, 1), raw=True).droplevel(0)

        return col_name

    @staticmethod
    def ts_shift(df, fcol, n, n_sup=7):
        """计算n日滞后值
        对于较大的n值, 进行shift容易overfitting, 因此限制n最大为7 (质数)
        若需去掉此限制，n_sup = -1
        """
        n = abs(n) # 防止forward looking
        if n_sup > 0:
            n %= n_sup
            n += 1 if n == 0 else 0

        col_name = 'ts_shift(%s,%s)' % (fcol, n)
        if col_name not in df.columns:
            df[col_name] = df.groupby('SecuCode')[fcol].shift(n)

        return col_name

    @staticmethod
    def ts_diff(df, fcol, n, n_sup=7):
        """计算n日差分
        对于较大的n值, 进行差分容易overfitting, 因此限制n最大为7 (质数)
        若需去掉此限制，n_sup = -1
        """
        n = abs(n) # 防止forward looking
        if n_sup > 0:
            n %= n_sup
            n += 1 if n == 0 else 0
        col_name = 'ts_diff(%s,%s)' % (fcol, n)
        if col_name not in df.columns:
            df[col_name] = df.groupby('SecuCode')[fcol].diff(n)

        return col_name

    @staticmethod
    def ts_median_abs_deviation(df, fcol, n=20):
        """n日中序列和均值距离的中位数 median(abs(x - x.mean()))
        """
        if n < 5:
            n += 5

        col_name = 'ts_median_abs_deviation(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.median_abs_deviation, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_rms(df, fcol, n=20):
        """n日均方根值 sqrt(mean(x**2))
        """
        if n < 5:
            n += 5

        col_name = 'ts_rms(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.rms, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_norm_mean(df, fcol, n=20):
        """ n日均值 / n日均方根值
        """
        if n < 5:
            n += 5

        col_name = 'ts_norm_mean(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.norm_mean, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_norm_max(df, fcol, n=20):
        """ n日最大值 / n日均方根值
        """
        if n < 5:
            n += 5

        col_name = 'ts_norm_max(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.norm_max, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_norm_min(df, fcol, n=20):
        """ n日最小值 / n日均方根值
        """
        if n < 5:
            n += 5

        col_name = 'ts_norm_min(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.norm_min, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_norm_min_max(df, fcol, n=20):
        """ n日最大值 - n日最小值 / n日均方根值
        """
        if n < 5:
            n += 5

        col_name = 'ts_norm_min_max(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.norm_min_max, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_ratio_beyond_3sigma(df, fcol, n=20):
        """
        超出3sigma 的比例
        """
        if n < 10:
            n += 10

        @numba.jit(nopython=True)
        def ratio_beyond_r_sigma(x, r=3):

            return np.mean(np.abs(x-np.mean(x)) > r * np.std(x))
        col_name = 'ts_ratio_beyond_3sigma(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(ratio_beyond_r_sigma, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_ratio_beyond_2sigma(df, fcol, n=20):
        """
        超出2sigma 的比例
        """
        if n < 10:
            n += 10

        @numba.jit(nopython=True)
        def ratio_beyond_r_sigma(x, r=2):

            return np.mean(np.abs(x-np.mean(x)) > r * np.std(x))
        col_name = 'ts_ratio_beyond_2sigma(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(ratio_beyond_r_sigma, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_index_mass_median(df, fcol, n=20):
        """
        返回index, 在过去n天里, 有50% 比例的质量落在了index 左侧
        """
        if n < 10:
            n += 10
        col_name = 'ts_index_mass_median(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.index_mass_quantile, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_number_cross_mean(df, fcol, n=20):
        """
        x 穿过mean的次数
        """
        col_name = 'ts_number_cross_mean(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.number_cross_mean, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_time_asymmetry_stats(df, fcol, n=20):
        """
        Returns the time reversal asymmetry statistic.
        .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag}^2 \\cdot x_{i + lag} - x_{i + lag} \\cdot  x_{i}^2
        """
        if n < 10:
            n += 10
        col_name = 'ts_time_asymmetry_stats(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[
            fcol].rolling(n).apply(NumbaFuncs.time_asymmetry_stats, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_longest_strike_above_mean(df, fcol, n=20):
        """
        高于均值的最长时间
        """
        if n < 10:
            n += 10
        col_name = 'ts_longest_strike_above_mean(%s,%s)' % (fcol, n)
        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.longest_steike_above_mean, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_longest_strike_below_mean(df, fcol, n=20):
        """
        低于均值的最长时间
        """
        if n < 10:
            n += 10
        col_name = 'ts_longest_strike_below_mean(%s,%s)' % (fcol, n)

        df[col_name] = df.groupby('SecuCode')[fcol].rolling(
            n).apply(NumbaFuncs.longest_steike_below_mean, raw=True).reset_index(level=0, drop=True)

        return col_name

    @staticmethod
    def ts_mean_over_1norm(df, fcol, n=20):
        """mean(x) / mean(abs(x))
        """
        df[f'__mean({fcol})'] = df.groupby('SecuCode')[fcol].rolling(
            n).mean().reset_index(level=0, drop=True)
        df[f'__abs({fcol})'] = df[fcol].abs()
        df[f'__abs_mean({fcol})'] = df.groupby('SecuCode')[f'__abs({fcol})'].rolling(
            n).mean().reset_index(level=0, drop=True)
        col_name = 'ts_mean_over_1norm(%s,%s)' % (fcol, n)

        df[col_name] = df[f'__mean({fcol})'] / df[f'__abs_mean({fcol})']
        df.drop(columns=[f'__mean({fcol})', f'__abs({fcol})',
                f'__abs_mean({fcol})'], inplace=True, errors='ignore')

        return col_name

    @staticmethod
    def ts_norm(df, fcol, n=20):
        df[f'__mean({fcol})'] = df.groupby('SecuCode')[fcol].rolling(
            n).mean().reset_index(level=0, drop=True)
        df[f'__std({fcol})'] = df.groupby('SecuCode')[fcol].rolling(
            n).std().reset_index(level=0, drop=True)
        col_name = 'ts_norm(%s,%s)' % (fcol, n)
        df[col_name] = (
            (df[fcol] - df[f'__mean({fcol})']) / df[f'__std({fcol})']).clip(-3, 3)

        df[col_name] = df[col_name].replace(-np.inf, np.nan)
        df.drop(
            columns=[f'__mean({fcol})', f'__std({fcol})'], inplace=True, errors='ignore')

        return col_name


class SingleAttrOperator:
    """
    只需要一列自身进行变化就可以完成
    input df, fcol
    return: col_name
    """

    @staticmethod
    def cs_rank(df, fcol):
        col_name = f'cs_rank({fcol})'
        df[col_name] = df.groupby('TradingDay')[fcol].rank(pct=True)

        return col_name

    @staticmethod
    def cs_norm(df, fcol):
        col_name = f'cs_norm({fcol})'
        df[col_name] = df.groupby('TradingDay')[fcol].transform(
            lambda x: NumbaFuncs.normalize(x.values))

        return col_name

    @staticmethod
    def ind_neu(df, fcol, ind_col='SWF'):
        mean_col = f'__mean_cs_neutralize{(fcol)}'
        std_col = f'__std_cs_neutralize{fcol}'
        df[mean_col] = df.groupby(['TradingDay', ind_col])[fcol].transform('mean')
        df[std_col] = df.groupby(['TradingDay', ind_col])[fcol].transform('std')

        col_name = f"ind_neu({fcol})"
        df[col_name] = (df[fcol] - df[mean_col]) / df[std_col]

        df.drop(columns=[mean_col, std_col], errors='ignore', inplace=True)

        return col_name


class DoubleAttrOperator:
    """

    """

    @staticmethod
    def comb_add(df, col1, col2, comparable=True):
        """
        因子相加，如果不可比则取截面rank
        """
        if not comparable:
            col1 = SingleAttrOperator.cs_rank(df, col1)
            col2 = SingleAttrOperator.cs_rank(df, col2)
        col_name = f'comb_add({col1},{col2})'
        df[col_name] = df[col1] + df[col2]

        if not comparable:
            df.drop(columns=[col1, col2], inplace=True, errors='ignore')

        return col_name

    @staticmethod
    def comb_sub(df, col1, col2, comparable=True):
        """
        因子相减，如果不可比则取截面rank
        """
        if not comparable:
            col1 = SingleAttrOperator.cs_rank(df, col1)
            col2 = SingleAttrOperator.cs_rank(df, col2)
        col_name = f'comb_sub({col1},{col2})'
        df[col_name] = df[col1] - df[col2]

        if not comparable:
            df.drop(columns=[col1, col2], inplace=True, errors='ignore')

        return col_name

    @staticmethod
    def comb_mul(df, col1, col2):
        """
        因子相乘
        """

        col_name = f'comb_mul({col1},{col2})'
        df[col_name] = df[col1] * df[col2]
        df.groupby('TradingDay')[col_name].transform(
            lambda x: NumbaFuncs.qclip(x.values))

        return col_name

    @staticmethod
    def comb_div(df, col1, col2):
        """
        因子相除
        """

        col_name = f'comb_div({col1},{col2})'
        df[col_name] = df[col1] / df[col2]
        df.groupby('TradingDay')[col_name].transform(
            lambda x: NumbaFuncs.qclip(x.values))

        return col_name

    @staticmethod
    def ts_corr_10D(df, col1, col2):
        """
        计算较为复杂, 因此滚动窗口只取10天,20日,40日三种, 搜索时可以避免重复计算
        rolling correlation of col1 and col2

        """
        col_name = f'ts_corr_10D({col1},{col2})'
        tmp = df.groupby('TradingDay').rolling(
            10)[[col1, col2]].corr().droplevel([0, 2])

        df[col_name] = - tmp.reset_index().drop_duplicates(
            subset=['index'], keep='last').set_index('index').iloc[:, 0]

        return col_name

    @staticmethod
    def ts_corr_20D(df, col1, col2):
        """
        计算较为复杂, 因此滚动窗口只取10天,20日,40日三种, 搜索时可以避免重复计算
        rolling correlation of col1 and col2

        """
        col_name = f'ts_corr_20D({col1},{col2})'
        tmp = df.groupby('TradingDay').rolling(
            20)[[col1, col2]].corr().droplevel([0, 2])

        df[col_name] = - tmp.reset_index().drop_duplicates(
            subset=['index'], keep='last').set_index('index').iloc[:, 0]

        return col_name

    @staticmethod
    def ts_corr_40D(df, col1, col2):
        """
        计算较为复杂, 因此滚动窗口只取10天,20日,40日三种, 搜索时可以避免重复计算
        rolling correlation of col1 and col2

        """
        col_name = f'ts_corr_40D({col1},{col2})'
        tmp = df.groupby('TradingDay').rolling(
            40)[[col1, col2]].corr().droplevel([0, 2])

        df[col_name] = - tmp.reset_index().drop_duplicates(
            subset=['index'], keep='last').set_index('index').iloc[:, 0]

        return col_name
