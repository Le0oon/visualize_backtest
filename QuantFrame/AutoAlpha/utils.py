import numpy as np
import logging


def reduce_mem_usage(df, verbose=False, logger=None):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Check if the column's data type is a float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        decrease = 100 * (start_mem - end_mem) / start_mem
        if logger is None:
            print(f"Memory usage of dataframe is {start_mem:.2f} MB")
            print(f"Memory usage after optimization is: {end_mem:.2f} MB")
            print(f"Decreased by {decrease:.2f}%")
        else:
            logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
            logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
            logger.info(f"Decreased by {decrease:.2f}%")

    # Return the DataFrame with optimized memory usage
    return df
