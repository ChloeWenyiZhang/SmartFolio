import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import torch
print(torch.cuda.is_available())
from dataloader.data_loader import *
from policy.policy import *
# from trainer.trainer import *
from trainer.irl_trainer import *
from torch_geometric.loader import DataLoader

PATH_DATA = f'./dataset/'

def train_predict(args, predict_dt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    train_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                        train_start_date=args.train_start_date, train_end_date=args.train_end_date,
                                        mode="train")
    val_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                      val_start_date=args.val_start_date, val_end_date=args.val_end_date,
                                      mode="val")
    test_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                       test_start_date=args.test_start_date, test_end_date=args.test_end_date,
                                       mode="test")
    train_loader_all = DataLoader(train_dataset, batch_size=len(train_dataset), pin_memory=True, collate_fn=lambda x: x,
                                  drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)
    print(len(train_loader), len(val_loader), len(test_loader))

    # create model
    env_init = create_env_init(args, dataset=train_dataset)
    if args.policy == 'MLP':
        model = PPO(policy='MlpPolicy',
                    env=env_init,
                    **PPO_PARAMS,
                    seed=args.seed,
                    device=args.device)
    elif args.policy == 'HGAT':
        model = PPO(policy=HGATActorCriticPolicy,
                    env=env_init,
                    **PPO_PARAMS,
                    seed=args.seed,
                    device=args.device)
    train_model_and_predict(model, args, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transaction ..")
    parser.add_argument("-device", "-d", default="cuda:0", help="gpu")
    parser.add_argument("-model_name", "-nm", default="SmartFolio", help="模型名称")
    parser.add_argument("-market", "-mkt", default="hs300", help="股票市场")
    parser.add_argument("-horizon", "-hrz", default="1", help="预测距离")
    parser.add_argument("-relation_type", "-rt", default="hy", help="股票关系类型")
    parser.add_argument("-ind_yn", "-ind", default="y", help="是否加入行业关系图")
    parser.add_argument("-pos_yn", "-pos", default="y", help="是否加入动量关系图")
    parser.add_argument("-neg_yn", "-neg", default="y", help="是否加入反转关系图")
    parser.add_argument("-multi_reward_yn", "-mr", default="y", help="是否加入多奖励学习")
    parser.add_argument("-policy", "-p", default="MLP", help="策略网络")
    args = parser.parse_args()

    # debug 用参数设置
    args.model_name = 'SmartFolio'
    args.market = 'hs300'
    args.relation_type = 'hy'
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.train_start_date = '2019-01-02'
    args.train_end_date = '2022-12-30'
    args.val_start_date = '2023-01-03'
    args.val_end_date = '2023-12-29'
    args.test_start_date = '2024-01-02'
    args.test_end_date = '2024-12-30'
    args.batch_size = 32
    args.max_epochs = 60
    args.seed = 123
    args.input_dim = 6
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.multi_reward = True

    if args.market == 'hs300':
        args.num_stocks = 102
    elif args.market == 'zz500':
        args.num_stocks = 80
    elif args.market == 'nd100':
        args.num_stocks = 84
    elif args.market == 'sp500':
        args.num_stocks = 472

    train_predict(args, predict_dt='2025-02-05')

    print(1)




