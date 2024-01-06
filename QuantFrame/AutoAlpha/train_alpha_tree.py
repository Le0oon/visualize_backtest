
import distutils
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count
import sys
import pandas as pd
import argparse
import logging

# backtest_frame_path = '/home/qzliu/QuantFrame'
backtest_frame_path = '/home/liuqize/QuantFrame'

sys.path.append(backtest_frame_path)


def setup_logger(log_file_path='training.log', logger_name='logger', console_output=True):
    # 创建一个logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # 创建一个文件处理器，用于将日志写入文件
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console_output:
        # 创建一个控制台处理器，用于将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


try:
    import my_config as config
except:
    import config
    print('Failed to import module my_config.py, use default config file.')
try:
    from Data import StockData
    from AutoAlpha import NaivePopulation, AutoAlphaPopulation
except Exception as e:
    raise e


def __get_model(arg):
    name = arg.model
    name = name.lower()
    if name == 'naive':
        return NaivePopulation
    elif name == 'auto_alpha':
        return AutoAlphaPopulation
    else:
        raise ValueError(f'Unknown model name: {name}')


def __get_ret_col(ret_col, ind_neu=True, derisk=False):
    """
    ret_cols = ['vwap_return','rcc_return','vwap_return_derisked', 'rcc_return_derisked',
       'cs_ind_neu(vwap_return_derisked)', 'cs_ind_neu(rcc_return_derisked)',
       'cs_ind_neu(vwap_return)', 'cs_ind_neu(rcc_return)']
    """
    # print()
    if ret_col in ['vwap', 'vwap_return', 'cs_norm(vwap_return)', 'cs_norm_vwap_return']:
        ret_col = 'vwap_return_derisked' if derisk else 'vwap_return'
        ret_col = f'cs_ind_neu({ret_col})' if ind_neu else ret_col
    elif ret_col in ['rcc', 'rcc_return', 'raw_close_close', 'cs_norm(rcc_return)', 'cs_norm_rcc_return']:
        ret_col = 'rcc_return_derisked' if derisk else 'rcc_return'
        ret_col = f'cs_ind_neu({ret_col})' if ind_neu else ret_col
    return ret_col


def __parse_warm_start(warm_start):
    if warm_start == 'True':
        return True
    elif warm_start == 'False':
        return False
    else:
        return warm_start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='naive')
    parser.add_argument('--data_path', type=str, default='/data/user_home/liuqize/auto_alpha_data/train_df.feather')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--population_size', type=int, default=2000)
    parser.add_argument('--rtype', type=str, default='vwap')
    parser.add_argument('--derisk', type=lambda x: bool(distutils.util.strtobool(x)), default=True)
    parser.add_argument('--ind_neu', type=lambda x: bool(distutils.util.strtobool(x)), default=True)
    parser.add_argument('--save_dir', type=str, default='/home/liuqize/GA_results')
    parser.add_argument('--multi_process', type=lambda x: bool(distutils.util.strtobool(x)), default=True)
    parser.add_argument('--warm_start', type=__parse_warm_start, default='True')
    args = parser.parse_args()

    # 选择模型
    Population = __get_model(args)
    # 生成存储结果的文件夹
    now = datetime.now().strftime('%Y%m%d')
    ret_col = __get_ret_col(args.rtype, args.ind_neu, args.derisk)
    save_dir = os.path.join(
        args.save_dir,
        f"{args.model}_{now}_depth{args.max_depth}_epoch{args.epoch}_pop{args.population_size}_ret_{ret_col}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    logger = setup_logger(os.path.join(save_dir, "train.log"))
    logger.info(f"Logs saved to {save_dir}")
    train_df = pd.read_feather(args.data_path)
    atp = Population(population_size=args.population_size,
                     max_depth=args.max_depth,
                     df=train_df,
                     ret_col=ret_col,
                     epoch=args.epoch,
                     clear_all_after_iter=True,
                     reduce_return=False,
                     save_dir=save_dir,
                     logger=logger,
                     warm_start=args.warm_start)

    atp.fit(multi_process=True)
