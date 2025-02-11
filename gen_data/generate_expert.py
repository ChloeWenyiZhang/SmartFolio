import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


def generate_expert_strategy(returns,
                             industry_relation_matrix,
                             correlation_matrix,
                             top_k=0.1,
                             max_industry_ratio=0.3):
    """
    生成专家策略：基于行业关系矩阵的分散化、相关性控制和动量效应。
    :param returns: 历史收益率数据（DataFrame，列为股票代码，行为日期）。
    :param industry_relation_matrix: 行业关系矩阵（num_stocks × num_stocks，权重表示行业关联强度）。
    :param correlation_matrix: 股票相关性矩阵（DataFrame，列为股票代码，行为股票代码）。
    :param top_k: 选择前 top_k 的股票。
    :param max_industry_ratio: 每个行业集群的最大权重。
    :return: 专家策略（二进制数组，1 表示选择，0 表示不选择）。
    """
    num_stocks = len(returns)
    # Step 1: 按收益率降序排列，生成候选队列
    candidate_indices = np.argsort(-returns).tolist()  # 降序排列的索引列表
    target_k = int(num_stocks * top_k)  # 目标选股数量
    expert_actions = np.zeros(num_stocks, dtype=int)
    selected_stocks = []
    industry_counts = {}  # 记录每个行业的已选数量
    # Step 2: 动态遍历候选队列，直到选满或队列为空
    while len(selected_stocks) < target_k and candidate_indices:
        idx = candidate_indices.pop(0)  # 取出当前最高收益的候选
        # 动态定义当前股票的行业集群
        industry_cluster = np.where(industry_relation_matrix[idx] > 0)[0].tolist()
        industry_cluster.append(idx)  # 包含自身
        # 统计当前行业已选数量
        selected_in_cluster = sum(expert_actions[industry_cluster])
        max_allowed = int(target_k * max_industry_ratio)
        # 行业限制检查
        if selected_in_cluster >= max_allowed:
            continue  # 跳过该股票，继续下一个候选
        # 相关性检查
        if selected_stocks:
            avg_corr = correlation_matrix[idx, selected_stocks].mean()
            if avg_corr >= 0.5:
                continue  # 相关性过高，跳过
        # 选中该股票
        expert_actions[idx] = 1
        selected_stocks.append(idx)
        # 更新行业计数
        for stock in industry_cluster:
            industry_counts[stock] = industry_counts.get(stock, 0) + 1
    return expert_actions


def generate_expert_trajectories(args, dataset, num_trajectories=100):
    """
    生成专家轨迹（状态-动作对），直接使用预处理好的时序特征。
    :param args: 命令行参数（包含市场、行业分类等信息）。
    :param dataset: 数据集（每个样本已包含时序特征和相关性矩阵）。
    :param num_trajectories: 生成的轨迹数量。
    :return: 专家轨迹列表，每个轨迹为 (state, action) 的序列。
    """
    expert_trajectories = []
    industry_relation_matrix = load_industry_relation_matrix(args.market)  # 加载行业关系矩阵

    for _ in range(num_trajectories):
        # 随机选择一个数据点,每个数据点已包含完整的时序特征
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]

        # 提取时序特征和相关性矩阵
        time_series_features = data['ts_features'].numpy()  # 形状为 [股票数量, 时间步长, 特征维度]
        features = data['features'].numpy()  # 形状为 [股票数量, 特征维度]
        correlation_matrix = data['corr'].numpy()  # 形状为 [股票数量, 股票数量]
        ind_matrix = data['industry_matrix'].numpy()  # 形状为 [股票数量, 股票数量]
        pos_matrix = data['pos_matrix'].numpy()  # 形状为 [股票数量, 股票数量]
        neg_matrix = data['neg_matrix'].numpy()  # 形状为 [股票数量, 股票数量]

        # 从时序特征中提取收益率（假设最后一列为收益率）
        returns = data['labels'].numpy()  # 转换为 [时间步长, 1]

        # 生成专家动作
        expert_actions = generate_expert_strategy(
            returns=returns,
            industry_relation_matrix=industry_relation_matrix,
            correlation_matrix=correlation_matrix
        )

        state = features.squeeze()
        if args.ind_yn:
            state = np.concatenate([state, ind_matrix], axis=1)
        if args.pos_yn:
            state = np.concatenate([state, pos_matrix], axis=1)
        if args.neg_yn:
            state = np.concatenate([state, neg_matrix], axis=1)
        expert_trajectories.append((state, expert_actions))
    return expert_trajectories


def load_industry_relation_matrix(market):
    """
    加载行业关系矩阵。
    :param market: 市场名称（如 'hs300'）。
    :return: 行业关系矩阵（num_stocks × num_stocks）。
    """
    with open(f"dataset_default/data_train_predict_{market}/industry.npy", 'rb') as f:
        industry_relation_matrix = np.load(f)
    return industry_relation_matrix


def process_state(features):
    """
    处理状态特征（如标准化）。
    :param features: 原始特征数据（numpy数组）。
    :return: 处理后的状态特征。
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled


def save_expert_trajectories(trajectories, save_path):
    """
    保存专家轨迹到文件。
    :param trajectories: 专家轨迹列表。
    :param save_path: 保存路径。
    """
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)


def load_expert_trajectories(load_path):
    """
    从文件加载专家轨迹。
    :param load_path: 加载路径。
    :return: 专家轨迹列表。
    """
    with open(load_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


if __name__ == '__main__':
    # 测试参数：生成并保存专家轨迹
    class Args:
        market = 'hs300'
        input_dim = 6


    args = Args()
    from dataloader.data_loader import AllGraphDataSampler

    # 加载数据集
    data_dir = f'../dataset/data_train_predict_{args.market}/1_hy/'
    train_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                        train_start_date='2019-01-02', train_end_date='2022-12-30',
                                        mode="train")

    # 生成专家轨迹
    expert_trajectories = generate_expert_trajectories(args, train_dataset, num_trajectories=100)

    # 保存专家轨迹
    save_path = f'..dataset/expert_trajectories_{args.market}.pkl'
    save_expert_trajectories(expert_trajectories, save_path)
    print(f"专家轨迹已保存至 {save_path}")