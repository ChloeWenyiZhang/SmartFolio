import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class Transpose(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class MHGraphAttn(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
        super(MHGraphAttn, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads*out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.linear = nn.Linear(num_heads*out_features, num_heads*out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, require_weights=False):
        batch = inputs.shape[0]
        # avoid divide 0 error
        adj_mat[torch.all(adj_mat == 0, dim=-1)] = 1e-6
        support = torch.matmul(inputs, self.weight)
        support = support.reshape(batch, -1, self.num_heads, self.out_features).permute(dims=(0, 2, 1, 3))
        f_1 = torch.matmul(support, self.weight_u).reshape(batch, self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(batch, self.num_heads, -1, 1)
        logits = f_1 + f_2
        weight = self.leaky_relu(logits)    # (batch_size, num_heads, num_nodes, num_nodes)
        masked_weight = torch.matmul(weight, adj_mat.unsqueeze(1)).to_sparse()
        attn_weights = torch.sparse.softmax(masked_weight, dim=3).to_dense()
        support = torch.matmul(attn_weights, support)   # (batch_size,num_heads, num_nodes, hidden_size)
        support = support.permute(dims=(0, 2, 1, 3)).reshape(batch, -1, self.num_heads * self.out_features)
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        # support = torch.tanh(self.linear(torch.tanh(support)))      # (batch_size, num_nodes, hidden_size)
        support = torch.tanh(support)
        if require_weights:
            return support, attn_weights
        else:
            return support, None

class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=1, keepdim=True)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=2).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=2, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=2, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x

class HeteFusionAttn(Module):
    def __init__(self, in_features, hidden_size=128, act=nn.Tanh()):
        super(HeteFusionAttn, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_features, hidden_size),
                                     act,
                                     nn.Linear(hidden_size, 1, bias=False))

    def forward(self, inputs, require_weights=False):
        w = self.project(inputs)
        beta = torch.softmax(w, dim=1)
        if require_weights:
            return (beta * inputs).sum(1), beta
        else:
            return (beta * inputs).sum(1), None

class HGAT(nn.Module):
    def __init__(self, num_stocks, n_features=8, num_heads=8, hidden_dim=512, no_ind=False, no_neg=False):
        super(HGAT, self).__init__()
        self.num_heads = num_heads
        self.n_adj_mat = num_stocks
        self.in_features = n_features
        self.out_features = n_features
        self.hidden_dim = hidden_dim
        self.ind_gat = MHGraphAttn(
            in_features=hidden_dim, out_features=n_features, num_heads=num_heads)
        self.pos_gat = MHGraphAttn(
            in_features=hidden_dim, out_features=n_features, num_heads=num_heads)
        self.neg_gat = MHGraphAttn(
            in_features=hidden_dim, out_features=n_features, num_heads=num_heads)

        self.mlp_self1 = nn.Linear(n_features, hidden_dim)
        self.ind_mlp = nn.Linear(n_features*num_heads, hidden_dim)
        self.pos_mlp = nn.Linear(n_features * num_heads, hidden_dim)
        self.neg_mlp = nn.Linear(n_features*num_heads, hidden_dim)
        self.mlp_self2 = nn.Linear(hidden_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU()

        self.pn = PairNorm(mode='PN-SI')
        self.sem_gat = HeteFusionAttn(in_features=hidden_dim, hidden_size=hidden_dim, act=nn.Tanh())
        self.generator = nn.Sequential(
            Transpose(),
            nn.Linear(num_stocks * hidden_dim, num_stocks),
            nn.Sigmoid()
        )
        self.no_ind = no_ind
        self.no_neg = no_neg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)

    def forward(self, inputs, require_weights=False):
        batch = inputs.shape[0]

        inputs = inputs.reshape(batch, -1, self.n_adj_mat)
        ind_adj = inputs[:, :self.n_adj_mat, :]
        pos_adj = inputs[:, self.n_adj_mat:self.n_adj_mat * 2, :]
        neg_adj = inputs[:, self.n_adj_mat * 2:self.n_adj_mat * 3, :]
        inputs = inputs[:, self.n_adj_mat * 3:, :].permute(dims=(0, 2, 1))

        support = self.mlp_self1(inputs)
        ind_support, ind_attn_weights = self.ind_gat(support, ind_adj, require_weights)
        pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, require_weights)
        neg_support, neg_attn_weights = self.neg_gat(support, neg_adj, require_weights)
        support = self.mlp_self2(support)
        ind_support = self.ind_mlp(ind_support)
        pos_support = self.pos_mlp(neg_support)
        neg_support = self.neg_mlp(neg_support)
        all_embedding = torch.stack((support, ind_support, pos_support, neg_support),
                                    dim=1)
        if self.no_ind:
            all_embedding = torch.stack((support, neg_support), dim=1)
        if self.no_neg:
            all_embedding = torch.stack((support, ind_support), dim=1)
        all_embedding, sem_attn_weights = self.sem_gat(all_embedding, require_weights)     # (batch, num_nodes, 64)
        all_embedding = self.pn(all_embedding)
        if require_weights:
            # ind_degrees = torch.sum(ind_adj, dim=1)
            # neg_degrees = torch.sum(neg_adj, dim=1)
            # ind_attn_weights = pd.Series(ind_attn_weights.cpu().mean(dim=0).mean(dim=0).mean(dim=0).detach().numpy())
            # neg_attn_weights = pd.Series(neg_attn_weights.cpu().mean(dim=0).mean(dim=0).mean(dim=0).detach().numpy())
            # sem_attn_weights = pd.Series(sem_attn_weights.cpu().mean(dim=0).mean(dim=2).mean(dim=1).detach().numpy())
            # attn_weights = pd.DataFrame()
            # attn_weights['ind_degrees'] = ind_degrees.cpu().mean(dim=0).detach().numpy()
            # attn_weights['neg_degrees'] = neg_degrees.cpu().mean(dim=0).detach().numpy()
            # attn_weights['ind_weights'] = ind_attn_weights
            # attn_weights['neg_weights'] = neg_attn_weights
            # attn_weights['sem'] = sem_attn_weights
            # attn_weights.to_csv('./results/attn_weights.csv', index=False)
            return self.generator(all_embedding)
        return self.generator(all_embedding)