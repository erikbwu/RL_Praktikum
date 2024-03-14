from imitation.rewards.reward_nets import RewardNet
import gymnasium as gym
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool, MLP
from torch_geometric.transforms import GridSampling

from sofa_imitation.policy.PointNetActorCritic import SAModule, GlobalSAModule


class PointNetRewardNet(RewardNet):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        grid_size: float,
        inp_features_dim: int,
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

        self.mlp = MLP([1024, 512, 128], norm=None)
        self.grid_sampling = GridSampling(grid_size)  # 3 ~> 1100, 2.5 ~> 1500

    def forward(self, state, action, next_state, done):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)

        outputs = self.mlp(inputs_concat)
        assert outputs.shape == state.shape[:1]

        return outputs
