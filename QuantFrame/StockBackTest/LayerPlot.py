
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class LayerPlot:
    @staticmethod
    def concat_factor_return(factor_df, return_df):
        '''
        将索引为日期&列为股票代码的因子df和收益df,合并成为外层索引为日期&内层索引为股票代码的df
        '''
        f_df = factor_df.stack().to_frame()
        r_df = return_df.stack().to_frame()
        f_df.rename(columns={0: "factor"}, inplace=True)
        r_df.rename(columns={0: "return"}, inplace=True)
        # 处理掉空值
        res = f_df.join(r_df).dropna(how='any')
        res = res.reset_index()
        res.rename(columns={'level_0': "date",
                   'level_1': "code"}, inplace=True)

        def strToDatetime(x): return datetime.strptime(x, "%Y-%m-%d")
        res['date'] = res['date'].apply(strToDatetime)
        res = res.sort_values(by=['date', 'code']).set_index(['date', 'code'])
        return res

    @staticmethod
    def get_layer_returns(concat_df, groups=30):
        day_list = list(set(concat_df.index.get_level_values(level=0)))
        day_list.sort()  # set是无序的
        pct_df = pd.DataFrame(index=range(1, groups + 1 + 1))
        for day in day_list:
            df = concat_df.loc[day]
            df = df[df['factor'].notnull()].sort_values(by='factor')
            # 平均分组
            stock_group_num, residual = [0], len(df) - len(df)//groups * groups
            for i in range(0, groups):
                last_num = stock_group_num[-1]
                avg = len(df)//groups
                idx = last_num + avg
                if residual > 0:
                    idx = idx + 1
                    residual = residual - 1
                stock_group_num.append(idx)

            if len(df) > 0:
                pct_se_list = []
                returns = df['return'].values
                # 不知道为什么偶尔可能会有某个收益率为inf,inf视为0处理吧
                returns = np.array(
                    list(map(lambda x: 0 if abs(x) == np.inf else x, returns)))
                for i in range(groups):
                    r = returns[stock_group_num[i]: stock_group_num[i+1]]
                    pct_se_list.append(sum(r)/(len(r)))
                pct_se_list.append(sum(returns)/(len(returns)))
                pct_df[day] = pct_se_list

        pct_df = pct_df.T
        pct_df.rename(columns={groups + 1: 'market'}, inplace=True)
        for col in pct_df.columns[0:-1]:
            pct_df[col] = pct_df[col] - pct_df['market']
        return pct_df

    @staticmethod
    def get_layer_cumret(pct_df, groups):
        day_list = list(set(pct_df.index))
        day_list.sort()
        start_day = day_list[0] + timedelta(days=-1)
        day_list = [start_day] + day_list

        cumret_df = pd.DataFrame(index=range(1, groups + 1 + 1))
        cumret_df[start_day] = np.array([0]*(groups + 1))
        previous_cumret = np.array([0]*(groups + 1))

        for day in day_list[1:]:
            r = pct_df.loc[day].values
            cum_r = previous_cumret + r
            cumret_df[day] = cum_r
            previous_cumret = cum_r
        cumret_df = cumret_df.T
        cumret_df.rename(columns={groups + 1: 'market'}, inplace=True)
        cumret_df = cumret_df.dropna(how='all')
        return cumret_df[cumret_df.columns[0:-1]]

    @staticmethod
    def get_layer_plot(factor_df_path, return_df_list, groups, threshold, figure_path=''):
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

        factor_df = pd.read_csv(factor_df_path, index_col=0, engine='python')

        flag = False

        fig = plt.figure(figsize=(16, 6 * 4))
        ax1 = plt.subplot(4, 1, 1)
        colors = gen_colors(groups)
        concat_df = LayerPlot.concat_factor_return(
            factor_df, return_df_list[0])
        pct_df = LayerPlot.get_layer_returns(concat_df, groups=groups)
        cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
        cumret_df.plot(color=colors, ax=ax1)
        figname = 'cumRet:{}'.format(
            str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
        ax1.set_title('{} {}'.format('whole market', figname))
        if max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1]) >= threshold[0]:
            flag = True

        ax2 = plt.subplot(4, 1, 2)
        colors = gen_colors(groups)
        concat_df = LayerPlot.concat_factor_return(
            factor_df, return_df_list[1])
        pct_df = LayerPlot.get_layer_returns(concat_df, groups=groups)
        cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
        cumret_df.plot(color=colors, ax=ax2)
        figname = 'cumRet:{}'.format(
            str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
        ax2.set_title('{} {}'.format('HS300', figname))
        if max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1]) >= threshold[1]:
            flag = True

        ax3 = plt.subplot(4, 1, 3)
        colors = gen_colors(groups)
        concat_df = LayerPlot.concat_factor_return(
            factor_df, return_df_list[2])
        pct_df = LayerPlot.get_layer_returns(concat_df, groups=groups)
        cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
        cumret_df.plot(color=colors, ax=ax3)
        figname = 'cumRet:{}'.format(
            str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
        ax3.set_title('{} {}'.format('ZZ500', figname))
        if max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1]) >= threshold[2]:
            flag = True

        ax4 = plt.subplot(4, 1, 4)
        colors = gen_colors(groups)
        concat_df = LayerPlot.concat_factor_return(
            factor_df, return_df_list[3])
        pct_df = LayerPlot.get_layer_returns(concat_df, groups=groups)
        cumret_df = LayerPlot.get_layer_cumret(pct_df, groups)
        cumret_df.plot(color=colors, ax=ax4)
        figname = 'cumRet:{}'.format(
            str(max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1])))
        ax4.set_title('{} {}'.format('ZZ1000', figname))
        if max(cumret_df.iloc[-1].iloc[0], cumret_df.iloc[-1].iloc[-1]) >= threshold[3]:
            flag = True

        # ax.set_title()
        if flag == True:
            fig.savefig(figure_path)
        return 1

    @staticmethod
    def plot_all_figures(factor_path, return_path_list, fig_path, groups, threshold, max_process):
        factor_name_list = []
        for root, dirs, files in os.walk(factor_path):
            for file in files:
                if os.path.splitext(file)[1] == '.csv':
                    factor_name_list.append(os.path.splitext(file)[0])

        return_df_list = []
        for return_path in return_path_list:
            return_df = pd.read_csv(return_path, index_col=0)
            # 重铸索引
            return_df.index = return_df.index.values
            return_df_list.append(return_df)

        if max_process > 1:
            pool = Pool(max_process)
            for factor_name in factor_name_list:
                factor_df_path = '{}/{}.csv'.format(factor_path, factor_name)
                figure_path = '{}/{}.png'.format(fig_path, factor_name)
                # LayerPlot.get_layer_plot(factor_df, return_df_list, groups, figure_path)
                pool.apply_async(LayerPlot.get_layer_plot,
                                 (factor_df_path, return_df_list, groups, threshold, figure_path))

                print('{}/{}.csv finished'.format(factor_path, factor_name))

            pool.close()
            pool.join()

        else:
            for factor_name in factor_name_list:
                factor_df = pd.read_csv(
                    '{}/{}.csv'.format(factor_path, factor_name), index_col=0, engine='python')
                figure_path = '{}/{}.png'.format(fig_path, factor_name)
                LayerPlot.get_layer_plot(
                    factor_df, return_df_list, groups, threshold, figure_path)
                print('{}/{}.csv finished'.format(factor_path, factor_name))

        return 1
