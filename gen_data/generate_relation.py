import time
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas.tseries.offsets import MonthEnd

feature_cols = ['close', 'open', 'high', 'low', 'prev_close', 'volume']

def cal_pccs(x, y, n):
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def calculate_pccs(xs, yss, n):
    result = []
    for name in yss:
        ys = yss[name]
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            tmp_res.append(cal_pccs(x, y, n))
        result.append(tmp_res)
    return np.mean(result, axis=1)

def stock_cor_matrix(ref_dict, codes, n, processes=1):
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[code], ref_dict, n) for code in codes]
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
        output = [o.get() for o in results]
        data = np.stack(output)
        return pd.DataFrame(data=data, index=codes, columns=codes)
    data = np.zeros([len(codes), len(codes)])
    for i in tqdm(range(len(codes))):
        data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
    return pd.DataFrame(data=data, index=codes, columns=codes)

STOCK_DATA_PATH = f"../dataset/"

if __name__ == '__main__':
    # 2018-01-01 - 2024-12-12
    market = 'hs300'
    train_start_date = '2018-01-01'
    train_end_date = '2022-12-31'
    eval_start_date = '2023-01-01'
    eval_end_date = '2023-12-31'
    test_start_date = '2024-01-01'
    test_end_date = '2024-12-31'
    relation_type = 'hy'

    horizon_s = [1]
    lookback_s = [5, 10, 20]

    df1 = pd.read_csv(f'{STOCK_DATA_PATH}{market}_org.csv')
    prev_date_num = 20
    date_unique = df1['dt'].unique()
    stock_trade_data = date_unique.tolist()
    stock_trade_data.sort()
    stock_num = df1.kdcode.unique().shape[0]
    # dt is the last trading day of each month
    dt = []
    for i in ['2018', '2019', '2020', '2021', '2022', '2023', '2024']:
        for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
            dt.append(stock_m[-1])
    df1['dt'] = df1['dt'].astype('datetime64[ns]')

    for i in range(len(dt)):
        df2 = df1.copy()
        end_date = dt[i]
        start_date = stock_trade_data[stock_trade_data.index(end_date) - (prev_date_num - 1)]
        df2 = df2.loc[df2['dt'] <= end_date]
        df2 = df2.loc[df2['dt'] >= start_date]
        code = sorted(list(set(df2['kdcode'].values.tolist())))
        test_tmp = {}
        for j in tqdm(range(len(code))):
            df3 = df2.loc[df2['kdcode'] == code[j]]
            y = df3[feature_cols].values
            if y.T.shape[1] == prev_date_num:
                test_tmp[code[j]] = y.T
        t1 = time.time()
        result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1)
        result = result.fillna(0)
        for i in range(0, stock_num):
            result.iloc[i, i] = 1
        t2 = time.time()

        corr_path = f"../dataset/corr/{market}/"
        if not os.path.exists(corr_path):
            os.makedirs(corr_path)

        # 检查是否已经是该月的最后一天
        end_date = pd.to_datetime(end_date)
        if end_date.day == end_date.replace(day=1).days_in_month:
            relation_dt = end_date
        else:
            relation_dt = end_date + MonthEnd(1)    # 该月最后一天（非最后一个交易日）
        relation_dt = relation_dt.strftime('%Y-%m-%d')

        print(relation_dt)
        print('time cost', t2 - t1, 's')
        result.to_csv(corr_path + str(relation_dt) + ".csv")
