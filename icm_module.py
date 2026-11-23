import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(self, obs_shape, action_size, feature_dim=256, device='cpu'):
        super(ICM, self).__init__()
        self.device = device

        # Feature extractor network
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 60, 60)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 30, 30)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 15, 15)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 15 * 15, feature_dim),  # Adjusted to 128 channels
            nn.ReLU()
        )

        # Inverse model network
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),  # Increased layer size
            nn.ReLU(),
            nn.Linear(512, 256),  # Added an additional layer
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

        # Optimizer for the curiosity module (created here so
        # train_step() can update the inverse/forward models).
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Forward model network
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_size, 512),  # Increased layer size
            nn.ReLU(),
            nn.Linear(512, 256),  # Added an additional layer
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state, next_state, action):
        state_features = self.feature_extractor(state)
        next_state_features = self.feature_extractor(next_state)

        # Debug prints
        # print(f"state_features shape: {state_features.shape}")
        # print(f"next_state_features shape: {next_state_features.shape}")

        inverse_input = torch.cat([state_features, next_state_features], dim=1)
        pred_action = self.inverse_model(inverse_input)

        action_onehot = F.one_hot(action, num_classes=pred_action.shape[1]).float()
        forward_input = torch.cat([state_features, action_onehot], dim=1)
        pred_next_state_features = self.forward_model(forward_input)

        # Debug prints
        # print(f"pred_action shape: {pred_action.shape}")
        # print(f"pred_next_state_features shape: {pred_next_state_features.shape}")

        return pred_action, pred_next_state_features, next_state_features

    def compute_intrinsic_reward(self, state, next_state, action):
        _, pred_next_state_features, next_state_features = self.forward(state, next_state, action)
        intrinsic_reward = F.mse_loss(pred_next_state_features, next_state_features, reduction='none').mean(dim=1)

        # Debug prints
        # print(f"intrinsic_reward shape: {intrinsic_reward.shape}")

        return intrinsic_reward

    def train_step(self, state, next_state, action, beta=0.2):
        """One ICM update on a batch of transitions.

        Inverse loss (cross-entropy on the predicted action) + forward
        loss (MSE on the predicted next-state features). Without this,
        compute_intrinsic_reward() is just noise from a randomly
        initialised network that never learns.
        """
        pred_action, pred_next_state_features, next_state_features = self.forward(
            state, next_state, action
        )
        inverse_loss = F.cross_entropy(pred_action, action.long())
        forward_loss = F.mse_loss(pred_next_state_features, next_state_features)
        loss = (1 - beta) * inverse_loss + beta * forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "icm_loss": float(loss.detach()),
            "icm_inverse_loss": float(inverse_loss.detach()),
            "icm_forward_loss": float(forward_loss.detach()),
        }
