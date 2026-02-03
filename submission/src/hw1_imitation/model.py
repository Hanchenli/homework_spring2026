"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class ReLUSquared(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 256, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]), ReLUSquared(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), ReLUSquared(),
            nn.Linear(hidden_dims[1], hidden_dims[2]), ReLUSquared(),
            nn.Linear(hidden_dims[2], action_dim * chunk_size),
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_chunk = self.sample_actions(state, num_steps=1)
        loss = nn.MSELoss(reduction='mean')(pred_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        pred_chunk = self.layers(state)
        return pred_chunk.reshape(state.shape[0], self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        # Input: state + flattened noisy action + time t
        input_dim = state_dim + action_dim * chunk_size + 1
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), ReLUSquared(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), ReLUSquared(),
            nn.Linear(hidden_dims[1], hidden_dims[2]), ReLUSquared(),
            nn.Linear(hidden_dims[2], action_dim * chunk_size),
        )

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity given state, noisy action x_t, and time t."""
        # x_t: (batch, chunk_size, action_dim) -> flatten to (batch, chunk_size * action_dim)
        x_t_flat = x_t.reshape(x_t.shape[0], -1)
        # t: (batch,) -> (batch, 1)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.view(-1, 1)
        # Concatenate state, flattened noisy action, and time
        inputs = torch.cat([state, x_t_flat, t], dim=-1)
        velocity = self.layers(inputs)
        return velocity.reshape(x_t.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        # Sample random time t ~ U(0, 1)
        t = torch.rand(batch_size, device=state.device)
        # Sample noise from standard normal
        noise = torch.randn_like(action_chunk)
        # Interpolate: x_t = (1 - t) * noise + t * action
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * action_chunk
        # Target velocity is action - noise (direction from noise to data)
        target_velocity = action_chunk - noise
        # Predict velocity
        pred_velocity = self.forward(state, x_t, t)
        # MSE loss
        loss = nn.MSELoss(reduction='mean')(pred_velocity, target_velocity)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        # Start from noise at t=0
        x_t = torch.randn(batch_size, self.chunk_size, self.action_dim, device=state.device)
        dt = 1.0 / num_steps
        # Euler integration from t=0 to t=1
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=state.device)
            velocity = self.forward(state, x_t, t)
            x_t = x_t + velocity * dt
        return x_t


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
