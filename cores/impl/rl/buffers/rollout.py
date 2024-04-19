from __future__ import annotations

import torch
from torch_geometric.data.batch import Batch


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.experiences = None

    def clear(self) -> None:
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

        self.experiences = None

    def __len__(self) -> int:
        return len(self.actions)

    def append(
        self,
        reward: torch.FloatTensor = None,
        done: torch.BoolTensor = None,
        state: torch.FloatTensor = None,
        action: torch.FloatTensor = None,
        action_logprob: torch.FloatTensor = None,
        state_value: torch.FloatTensor = None,
    ) -> None:
        if reward is not None:  # Add reward to buffer
            self.rewards.append(reward)
        if done is not None:  # Add done to buffer
            self.is_terminals.append(done)
        if action is not None:  # Add action to buffer
            self.actions.append(action)
        if action_logprob is not None:  # Add action logprobs to buffer
            self.logprobs.append(action_logprob)
        if state is not None:  # Add state to buffer
            state_buff = state.clone()
            # delattr(state_buff, 'batch')
            self.states.append(state_buff)
        if state_value is not None:  # Add state to buffer
            self.state_values.append(state_value)

    def get(self, idx: int):
        action = self.actions[idx]
        state = self.states[idx]
        log_prob = self.logprobs[idx]
        reward = self.rewards[idx]
        done = self.is_terminals[idx]
        return action, state, log_prob, reward, done

    @torch.no_grad()
    def prepare(self, gamma: float, device: str = "cpu"):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:  # This if statement handles new episodes
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards_original = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = rewards_original/ (rewards_original.std() + 1e-7)
        rewards = (rewards_original - rewards_original.mean()) / (rewards_original.std() + 1e-7)
        rewards = rewards.detach()

        old_states = Batch.from_data_list(self.states).to(device)

        old_actions = torch.cat(self.actions).detach().to(device)

        old_logprobs = torch.cat(self.logprobs).detach().to(device)
        old_state_values = torch.cat(self.state_values).detach().to(device)
        if old_state_values.ndim == 2:
            assert old_state_values.shape[-1] == 1
            old_state_values = old_state_values.flatten()

        advantages = rewards - old_state_values
        experiences = [old_states, old_actions, old_logprobs, rewards, rewards_original, advantages]

        self.experiences = experiences

        return
