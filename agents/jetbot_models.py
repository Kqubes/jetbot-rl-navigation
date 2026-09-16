from __future__ import annotations

import torch
import torch.nn as nn

from tensordict import TensorDict
from rsl_rl.models.mlp_model import MLPModel


class JetBotTemporalVisionModel(MLPModel):
    """Temporal CNN model for JetBot vision-based navigation."""

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ):
        active_obs_groups = obs_groups[obs_set]
        obs_dim = sum(obs[group][0].numel() for group in active_obs_groups)

        return active_obs_groups, obs_dim

    def _get_latent_dim(self):
        return 256

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(512, 256),
        activation="relu",
        obs_normalization=False,
        distribution_cfg=None,
        cnns=None,
    ):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=[256],
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        if cnns is None:
            temporal_cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Flatten(),
            )

            fusion = nn.Sequential(
                nn.Linear(3075, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )

            self.cnns = nn.ModuleList([temporal_cnn, fusion])
        else:
            self.cnns = cnns

        self.output_head = nn.Linear(256, output_dim)
        self.is_actor = obs_set == "actor"

        self.mlp = nn.Identity()

    def get_latent(
        self,
        obs: TensorDict,
        masks=None,
        hidden_state=None,
    ):
        images = obs["images"]
        state_vec = obs["policy"]

        frame_0 = images[:, 0:3]
        frame_1 = images[:, 3:6]
        frame_2 = images[:, 6:9]

        cnn = self.cnns[0]
        fusion = self.cnns[1]

        feature_0 = cnn(frame_0)
        feature_1 = cnn(frame_1)
        feature_2 = cnn(frame_2)

        visual_features = torch.cat(
            [feature_0, feature_1, feature_2],
            dim=-1,
        )

        combined = torch.cat(
            [visual_features, state_vec],
            dim=-1,
        )

        return fusion(combined)

    def forward(
        self,
        obs: TensorDict,
        stochastic_output: bool = True,
        **kwargs,
    ):
        latent = self.get_latent(obs)
        output = self.output_head(latent)

        if self.is_actor:
            output = torch.tanh(output)

        if self.distribution is not None:
            self.distribution.update(output)

            if stochastic_output:
                return self.distribution.sample()

            return self.distribution.deterministic_output(output)

        return output