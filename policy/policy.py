import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

from model import HGAT


class HGATNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
            n_head=8,
            hidden_dim=128,
            no_ind=False,
            no_neg=False,
    ):
        super(HGATNetwork, self).__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.num_stocks = last_layer_dim_pi
        self.n_features = feature_dim - self.num_stocks * 2

        self.policy_net = HGAT(num_stocks=self.num_stocks, n_features=self.n_features,
                               num_heads=n_head, hidden_dim=hidden_dim,
                               no_ind=no_ind, no_neg=no_neg)
        self.value_net = HGAT(num_stocks=self.num_stocks, n_features=self.n_features,
                              num_heads=n_head, hidden_dim=hidden_dim,
                              no_ind=no_ind, no_neg=no_neg)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            :return: (th.Tensor, th.Tensor) latent_policy, latent_value of     the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
            """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class HGATActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable[[float], float],
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 *args,
                 **kwargs,
                 ):
        super(HGATActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HGATNetwork(last_layer_dim_pi=self.action_space.shape[0],
                                         last_layer_dim_vf=self.action_space.shape[0],
                                         feature_dim=self.observation_space.shape[0],
                                         )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions, values, log_prob = super().forward(obs, deterministic)
        return actions, values, log_prob

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        actions, values, log_prob = self.forward(observation, deterministic)
        return actions
