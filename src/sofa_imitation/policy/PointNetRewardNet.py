import numpy as np
from torch_geometric.data import Data

from imitation.rewards.reward_nets import RewardNet
import gymnasium as gym
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool, MLP
from torch_geometric.transforms import GridSampling
import torch

from .PointNetActorCritic import SAModule, GlobalSAModule


class PointNetRewardNet(RewardNet):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        grid_size: float,
        inp_features_dim: int,
        action_features_dim: int,
        **kwargs,
    ):
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space, action_space)
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + inp_features_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp_points = MLP([1024, 512, 128], norm=None)
        self.mlp_global = MLP([128 + action_features_dim, 64, 32, 1], norm=None)
        self.grid_sampling = GridSampling(grid_size)  # 3 ~> 1100, 2.5 ~> 1500

    def forward(self, state, action, next_state, done):

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
            if len(state.shape) == 3 and state.shape[0] == 1:  # has batch dimension
                state = state[0]

            assert len(state.shape) == 2  # only handles one array observation
            if state.shape[-1] != 3:  # has color
                state = Data(pos=state[:, :3], batch=torch.full((len(state),), 0),
                                    x=state[:, 3:]).to(state.device)
            else:
                state = Data(pos=state, batch=torch.full((len(state),), 0)).to(
                    state.device)
            # display_array(state.pos.cpu(), state.x.cpu())
        if len(state.pos) > 5000:
            state = self.grid_sampling(state)
            # display_array(state.pos.cpu(), state.x.cpu())
        sa0_out = (state.x.to(torch.float32), state.pos.to(torch.float32), state.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        features = torch.cat([self.mlp_points(x), torch.tensor(action).to(self.device)], dim=1)
        #print(features.shape)
        outputs = self.mlp_global(features)

        return outputs

    def predict(
        self,
        state,
        action: np.ndarray,
        next_state,
        done,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.

        Converting th.Tensor rewards from `predict_th` to NumPy arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,)`.
        """
        rew_th = self(state, action, next_state, done)
        return rew_th.detach().cpu().numpy().flatten()
