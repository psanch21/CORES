from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym
import torch
import torch_geometric.data as pygd
import torch_geometric.loader as pygl

import cores.core.values.constants as Cte
from cores.impl.gnn.pyg.gnn_base import BaseGNN
from cores.impl.rl.graph.rewards import Reward


class GraphEnvBase(gym.Env, ABC):
    """Removes one node at a time"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        loader: pygl.DataLoader,
        graph_clf: BaseGNN,
        reward_fn: Reward,
        action_refers_to: Cte.ActionTypes = None,
        penalty_size: float = 0.0,
        max_episode_length: int = 100000,
        use_intrinsic_reward: int = False,
        device: int = "cpu",
    ):
        """

        Args:
            loader:
            graph_clf:
            reward_fn:
            desired_perc_nodes:
            max_episode_length: This is positive integer that controls the maximum number
            of steps an episode can have
            use_intrinsic_reward: This is a boolean that controls if we use intermediate
            (intrinsic) rewards  or not
            device:
        """
        super(GraphEnvBase, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        assert loader.batch_size == 1
        assert penalty_size >= 0
        assert not use_intrinsic_reward
        assert isinstance(action_refers_to, Cte.ActionTypes)
        # Example for using image as input:
        self.observation_space = None
        self.loader = loader

        self.graph_clf = graph_clf
        self.reward_fn = reward_fn
        self.state = None
        self.current_iter = 0
        self.current_reward = 0
        self.max_episode_length = max_episode_length
        self.use_intrinsic_reward = use_intrinsic_reward
        self.penalty_size = penalty_size
        self.action_refers_to = action_refers_to

        self._node_feature = "x"
        self._edge_attr = None

        self.init_num_nodes = None
        self.init_num_edges = None

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _is_empty(self, data: pygd.Data) -> bool:
        if self.action_refers_to == Cte.ActionTypes.EDGE:
            return data.edge_index.shape[1] == 0
        elif self.action_refers_to == Cte.ActionTypes.NODE:
            return data.x.shape[0] == 0

    @abstractmethod
    def _step(self, action: torch.Tensor, relabel_nodes: bool = True):
        pass

    def step(self, action: torch.Tensor, relabel_nodes: bool = True):
        self.validate_action(action)
        self.current_iter += 1
        return self._step(action, relabel_nodes)

    def validate_action(self, action: torch.Tensor):
        num_elements = self._num_elements(self.state)
        assert len(action) == num_elements, f"Action size {len(action)} != {num_elements}"

    @abstractmethod
    def get_action_distr_name(self):
        pass

    def get_final_graph(self) -> pygd.Data:
        return self.state

    def on_episode_end(self) -> None:
        if self.is_training:
            self.graph_clf.train()
        else:
            self.graph_clf.eval()

    def _is_original(self, data: pygd.Data) -> bool:
        return self._num_elements(data) == self.init_num_elements

    @property
    def init_num_elements(self) -> int:
        if self.action_refers_to == Cte.ActionTypes.EDGE:
            return self.init_num_edges
        elif self.action_refers_to == Cte.ActionTypes.NODE:
            return self.init_num_nodes

    def _num_elements(self, data: pygd.Data) -> int:
        if self.action_refers_to == Cte.ActionTypes.EDGE:
            return data.edge_index.shape[1]
        elif self.action_refers_to == Cte.ActionTypes.NODE:
            return data.x.shape[0]

    def reset(self, graph: Optional[pygd.Data] = None, idx: Optional[int] = None):
        # Reset the state of the environment to an initial state
        self.is_training = self.graph_clf.training
        self.graph_clf.eval()
        del self.state
        if graph is not None:
            state = graph.clone()
            state.batch = torch.zeros(
                state[self._node_feature].shape[0], dtype=torch.int64, device=self.device
            )
        elif idx is not None:
            data = self.loader.dataset.__getitem__(idx)
            state = copy.deepcopy(pygd.Batch.from_data_list([data]))

        else:
            state = copy.deepcopy(next(iter(self.loader)))
        self.state = state.clone().to(self.device)

        self.reward_fn.set_state_0(self.state.clone())

        self.current_iter = 0
        self.current_reward = 0
        self.init_num_nodes = self.state[self._node_feature].shape[0]
        self.init_num_edges = self.state.edge_index.shape[1]
        self.init_reward = self.reward_fn.compute(state=self.state, action=None).item()
        return self.state

    def render(self, mode: str = "human", close: bool = False):
        # Render the environment to the screen
        print(f"\nNumber of nodes: {self.state[self._node_feature].shape[0]}")
        print(f"Iteration: {self.current_iter}")
        print(f"Reward: {self.current_reward}")
        return

    def __str__(self) -> str:
        my_str = "\nGraph Environment"
        my_str += f"\n\tMax episode length: {self.max_episode_length}"
        my_str += f"\n\tUse intrinsic reward: {self.use_intrinsic_reward}"
        my_str += f"\n\tNumber of graphs in the environment: {len(self.loader)}"
        return my_str
