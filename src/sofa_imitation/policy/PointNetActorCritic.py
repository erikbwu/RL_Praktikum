import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool, MLP


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.r = r
        self.ratio = ratio
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, features_dim], norm=None)

    def forward(self, observations: Data) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            # num_points = torch.full((observations.shape[0],), 1)
            # batch = torch.repeat_interleave(
            #     torch.arange(len(num_points), device=num_points.device),
            #     repeats=num_points,
            # )
            # flattened_observations = torch.flatten(observations, end_dim=1)
            observations = Data(pos=observations, batch=torch.full((len(observations),), 0)).to(observations.device)

        sa0_out = (None, observations.pos.to(torch.float32), observations.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x).log_softmax(dim=-1)


# class PointNetActorCriticPolicy(ActorCriticPolicy):
#     def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
#         super(PointNetActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
#
#         with tf.variable_scope("model", reuse=reuse):
#             activ = tf.nn.relu
#
#             extracted_features = nature_cnn(self.processed_obs, **kwargs)
#             extracted_features = tf.layers.flatten(extracted_features)
#
#             pi_h = extracted_features
#             for i, layer_size in enumerate([128, 128, 128]):
#                 pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
#             pi_latent = pi_h
#
#             vf_h = extracted_features
#             for i, layer_size in enumerate([32, 32]):
#                 vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
#             value_fn = tf.layers.dense(vf_h, 1, name='vf')
#             vf_latent = vf_h
#
#             self._proba_distribution, self._policy, self.q_value = \
#                 self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
#
#         self._value_fn = value_fn
#         self._setup_init()
#
#     def step(self, obs, state=None, mask=None, deterministic=False):
#         if deterministic:
#             action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
#                                                    {self.obs_ph: obs})
#         else:
#             action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
#                                                    {self.obs_ph: obs})
#         return action, value, self.initial_state, neglogp
#
#     def proba_step(self, obs, state=None, mask=None):
#         return self.sess.run(self.policy_proba, {self.obs_ph: obs})
#
#     def value(self, obs, state=None, mask=None):