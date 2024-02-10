import warnings
from typing import Callable, Optional, Union, Tuple, List, Dict, Type

import torch
import torch as th
import numpy as np
from gymnasium import spaces

#from processing.view_pointcloud import display_array
from torch_geometric.transforms import GridSampling

from stable_baselines3.common.policies import ActorCriticPolicy, SelfBaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
        self.grid_sampling = GridSampling(size=2.5)  #3 ~> 1100, 2.5 ~> 1500

    def forward(self, observations: Data) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            # num_points = torch.full((observations.shape[0],), 1)
            # batch = torch.repeat_interleave(
            #     torch.arange(len(num_points), device=num_points.device),
            #     repeats=num_points,
            # )
            # flattened_observations = torch.flatten(observations, end_dim=1)
            if len(observations.shape) == 2:  # only handles one array observation
                if observations.shape[-1] != 3: # has color
                    observations = Data(pos=observations[:, :3], batch=torch.full((len(observations),), 0),
                                        x=observations[:, 3:]).to(observations.device)
                else:
                    observations = Data(pos=observations, batch=torch.full((len(observations),), 0)).to(
                        observations.device)
        observations = self.grid_sampling(observations)
        #display_array(observations.pos.cpu(), observations.x.cpu())
        sa0_out = (None, observations.pos.to(torch.float32), observations.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x).log_softmax(dim=-1)


class PointNetActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            *args,
            **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            features_extractor_class=PointNetFeaturesExtractor,
            *args,
            **kwargs,
        )

    def extract_features(  # type: ignore[override]
            self, obs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return self.features_extractor(obs)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = observation, False  # todo changed

        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low,
                                  self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]

    def get_distribution(self, obs):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        obs = th.tensor(obs).to(self.device)
        features = self.pi_features_extractor(obs)  # super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def load(cls: Type[SelfBaseModel], path: str, device: Union[th.device, str] = "auto") -> SelfBaseModel:
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = th.device(device)

        # Cuda not available
        if device.type == th.device("cuda").type and not th.cuda.is_available():
            device = th.device("cpu")

        saved_variables = th.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model
