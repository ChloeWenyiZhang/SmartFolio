import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.portfolio_env import *

def mse_loss(logits, targets):
    mse = nn.MSELoss()
    loss = mse(logits.squeeze(), targets)
    return loss

def bce_loss(logits, targets):
    bce = nn.BCELoss()
    loss = bce(logits.squeeze(), targets)
    return loss

def process_data(data_dict, device="cuda:0"):
    corr = data_dict['corr'].to(device).squeeze()
    ts_features = data_dict['ts_features'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    pyg_data = data_dict['pyg_data'].to(device)
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask']
    return corr, ts_features, features, labels, pyg_data, mask

PPO_PARAMS = {
        "n_steps": 1024,
        "ent_coef": 0.005,
        "learning_rate": 1e-4,
        "batch_size": 128,
        "gamma": 0.5,
        "tensorboard_log": "./logs",
    }

def train_model_one(args, train_loader):
    for batch_idx, data in enumerate(train_loader):
        corr, ts_features, features, labels, pyg_data, mask = process_data(data, device=args.device)
        env_train = StockPortfolioEnv(args, corr, ts_features, features, labels, pyg_data)
        env_train.seed(seed=args.seed)
        env_train, _ = env_train.get_sb_env()
        if args.policy == 'MLP':
            model = PPO(policy='MlpPolicy',
                        env=env_train,
                        **PPO_PARAMS,
                        seed=args.seed,
                        device='cuda:0')
        trained_model = model.learn(total_timesteps=1000)
        return trained_model


# 用于创建占位环境，后续使用model.set_env()进行更新
def create_env_init(args, dataset=None, data_loader=None):
    if data_loader is None:
        data_loader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True, collate_fn=lambda x: x,
                                 drop_last=True)
    for batch_idx, data in enumerate(data_loader):
        corr, ts_features, features, labels, pyg_data, mask = process_data(data, device=args.device)
        env = StockPortfolioEnv(args, corr, ts_features, features, labels, pyg_data)
        env.seed(seed=args.seed)
        env, _ = env.get_sb_env()
        print("占位环境创建完成")
        return env


# train切分batch进行train
def train_model_and_predict(model, args, train_loader, val_loader, test_loader):
    env_val = create_env_init(args, data_loader=val_loader)
    for i in range(args.max_epochs):
        epoch_reward = 0.0
        for batch_idx, data in enumerate(train_loader):
            corr, ts_features, features, labels, pyg_data, mask = process_data(data, device=args.device)
            env_train = StockPortfolioEnv(args, corr, ts_features, features, labels, pyg_data)
            env_train.seed(seed=args.seed)
            env_train, _ = env_train.get_sb_env()
            model.set_env(env_train)
            trained_model = model.learn(total_timesteps=1000)
            batch_reward, std = evaluate_policy(model, env_train, n_eval_episodes=1)
            epoch_reward += batch_reward
        mean_reward, std_reward = evaluate_policy(model, env_val, n_eval_episodes=1)
        print(f"Epoch total reward: {epoch_reward},"
              f" Val 平均奖励: {mean_reward}, 标准差: {std_reward}")
        model_predict(args, trained_model, test_loader)


def train_model_and_predict_hierarchical(model, args, train_loader, val_loader, test_loader):
    env_val = create_env_init(args, data_loader=val_loader)
    for i in range(args.max_epochs):
        epoch_reward = 0.0
        for batch_idx, data in enumerate(train_loader):
            corr, ts_features, features, labels, pyg_data, mask = process_data(data, device=args.device)
            env_train = StockPortfolioEnv(args, corr, ts_features, features, labels, pyg_data)
            env_train.seed(seed=args.seed)
            env_train, _ = env_train.get_sb_env()
            model.set_env(env_train)
            trained_model = model.learn(total_timesteps=1000)
            batch_reward, std = evaluate_policy(model, env_train, n_eval_episodes=1)
            epoch_reward += batch_reward
        mean_reward, std_reward = evaluate_policy(model, env_val, n_eval_episodes=1)
        print(f"Epoch total reward: {epoch_reward},"
              f" Val 平均奖励: {mean_reward}, 标准差: {std_reward}")
        model_predict(args, trained_model, test_loader)

# 使用一整个train的时序作为一个batch
def train_and_predict(args, train_loader, test_loader):
    for batch_idx, data in enumerate(train_loader):
        corr, ts_features, features, labels, pyg_data, mask = process_data(data, device=args.device)
        env_train = StockPortfolioEnv(args, corr, ts_features, features, labels, pyg_data)
        env_train.seed(seed=args.seed)
        env_train, _ = env_train.get_sb_env()
        if args.policy == 'MLP':
            model = PPO(policy='MlpPolicy',
                        env=env_train,
                        **PPO_PARAMS,
                        seed=args.seed,
                        device='cuda:0')
        for i in range(args.max_epochs):
            trained_model = model.learn(total_timesteps=1000)
            # 评估训练后的模型
            mean_reward, std_reward = evaluate_policy(model, env_train, n_eval_episodes=1)
            print(f"平均奖励: {mean_reward}, 标准差: {std_reward}")
            model_predict(args, trained_model, test_loader)
        return trained_model


def model_predict(args, model, test_loader):
    # 读取指数 benchmark 数据，用于计算信息系数 IR
    df_benchmark = pd.read_csv(f"../dataset/index_data/{args.market}_index.csv")
    df_benchmark = df_benchmark[(df_benchmark['date'] >= args.test_start_date) &
                                (df_benchmark['date'] <= args.test_end_date)]
    benchmark_return = df_benchmark['return']
    for batch_idx, data in enumerate(test_loader):
        corr, ts_features, features, labels, pyg_data, mask = process_data(data, device=args.device)
        env_test = StockPortfolioEnv(args, corr, ts_features, features, labels, pyg_data, benchmark_return,
                                     mode="test")
        env_test, obs_test = env_test.get_sb_env()
        env_test.reset()
        max_step = len(labels)
        for i in range(max_step):
            action, _states = model.predict(obs_test)
            obs_test, rewards, dones, info = env_test.step(action)
            if dones[0]:
                print("测试结束！")
                break

