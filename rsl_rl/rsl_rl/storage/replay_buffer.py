import torch
import numpy as np

from math import floor
from dha.data.DhaDynamicsRecording import DhaDynamicsRecording


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, action_dim, buffer_size, device="cpu"):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.actions = torch.zeros(buffer_size, action_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states, actions, next_states):
        """Add new states to memory."""

        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            # add states to the end of the buffer and wrap around
            self.states[self.step:self.buffer_size] = states[:self.buffer_size - self.step]
            self.actions[self.step:self.buffer_size] = actions[:self.buffer_size - self.step]
            self.next_states[self.step:self.buffer_size] = next_states[:self.buffer_size - self.step]
            self.states[:end_idx - self.buffer_size] = states[self.buffer_size - self.step:]
            self.actions[:end_idx - self.buffer_size] = actions[self.buffer_size - self.step:]
            self.next_states[:end_idx - self.buffer_size] = next_states[self.buffer_size - self.step:]
        else:
            self.states[start_idx:end_idx] = states
            self.actions[start_idx:end_idx] = actions
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batches, mini_batch_size):
        for _ in range(num_mini_batches):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device),
                   self.actions[sample_idxs].to(self.device),
                   self.next_states[sample_idxs].to(self.device))

    def shape_states_actions(self, batch_states, batch_actions, batch_next_states):
        """Preprocess the samples if needed."""
        flat_sample = DhaDynamicsRecording.map_state_action_state(
            sample={
                "state_observations": batch_states.cpu().numpy(),
                "action_observations": batch_actions.cpu().numpy(),
                "next_state_observations": batch_next_states.cpu().numpy()
            },
            state_observations=["state_observations"],
            action_observations=["action_observations"],
        )

        # Convert numpy arrays in flat_sample to torch tensors
        for key, value in flat_sample.items():
            if isinstance(value, np.ndarray):
                flat_sample[key] = torch.from_numpy(value).to(self.device)

        return flat_sample

    @staticmethod
    def preprocess_samples(
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        frames_per_step: int = 1,
        prediction_horizon: int = 1,
    ):
        """Generator that yields observation samples of length `n_frames_per_state` from the Markov Dynamics recordings.

        Args:
            states (torch.Tensor): The states of the trajectories, shape [T, obs_dim].
            actions (torch.Tensor): The actions of the trajectories, shape [T, action_dim].
            next_states (torch.Tensor): The next states of the trajectories, shape [T, obs_dim].
            frames_per_step: Number of frames to compose a single observation sample at time `t`. E.g. if `f` is
            provided
            the state samples will be of shape [f, obs_dim].
            prediction_horizon (int, float): Number of future time steps to include in the next time samples.
                E.g: if `n` is an integer the samples will be of shape [n, frames_per_state, obs_dim]
                If `n` is a float, then the samples will be of shape [int(n*traj_length), frames_per_state, obs_dim]

        Returns:
            A dictionary containing the observations of shape (time_horizon, frames_per_step, obs_dim)
        """
        traj_length = states.shape[0]
        steps_in_pred_horizon = prediction_horizon
        assert steps_in_pred_horizon > 0, f"Invalid prediction horizon {steps_in_pred_horizon}"

        remnant = traj_length % frames_per_step
        frames_in_pred_horizon = steps_in_pred_horizon * frames_per_step
        # Iterate over the frames of the trajectory
        for frame in range(traj_length - frames_per_step):
            # Collect the next steps until the end of the trajectory
            if frame + frames_per_step + frames_in_pred_horizon > (traj_length - remnant):
                continue
            sample = {}
            num_steps = (frames_in_pred_horizon // frames_per_step) + 1
            # Compute the indices for the start and end of each "step" in the time horizon
            start_indices = np.arange(0, num_steps) * frames_per_step + frame
            end_indices = start_indices + frames_per_step
            # Use these indices to slice the relevant portion of the trajectory
            obs_time_horizon = states[start_indices[0] : end_indices[-1]]
            next_obs_time_horizon = next_states[start_indices[0] : end_indices[-1]]
            action_time_horizon = actions[start_indices[0] : end_indices[-1]]
            # Reshape the slice to have the desired shape (time, frames_per_step, obs_dim)
            obs_time_horizon = obs_time_horizon.reshape((num_steps, frames_per_step, states.shape[1]))
            next_obs_time_horizon = next_obs_time_horizon.reshape((num_steps, frames_per_step, next_states.shape[1]))
            action_time_horizon = action_time_horizon.reshape((num_steps, frames_per_step, actions.shape[1]))

            sample["state_observations"] = obs_time_horizon
            sample["action_observations"] = action_time_horizon
            sample["next_state_observations"] = next_obs_time_horizon
            yield sample

class RunningStdScaler:
    """
    A class to compute a running standard deviation and apply scaling
    to input observations, without subtracting the mean.
    The formula applied is: x' = x / std(X)
    """
    def __init__(self, state_dim, action_dim, device):
        """
        Initializes the RunningStdScaler.

        Args:
            state_dim (int): The dimension of the states.
            action_dim (int): The dimension of the actions.
            device (torch.device): The device (e.g., 'cpu' or 'cuda') to store tensors.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Initialize second moment (sum of squares) for variance calculation
        self.sum_sq_states = torch.zeros(self.state_dim, device=device)
        self.sum_sq_actions = torch.zeros(self.action_dim, device=device)
        self.count = 0.0
        self.epsilon = 1e-8 # Small value to prevent division by zero for std

    def update(self, states, actions):
        """
        Updates the running statistics (sum of squares and count) with a new batch of observations.

        Args:
            states (torch.Tensor): A batch of states. Expected shape (batch_size, state_dim).
            actions (torch.Tensor): A batch of actions. Expected shape (batch_size, action_dim).
        """
        # Ensure observations are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)

        batch_count = states.shape[0]

        # We need to maintain the sum of squares to compute variance.
        # This is a simplified online variance calculation for std only,
        # assuming the mean is implicitly zero or irrelevant for scaling.
        self.sum_sq_states += torch.sum(states**2, dim=0).to(self.device)
        self.sum_sq_actions += torch.sum(actions**2, dim=0).to(self.device)
        self.count += batch_count

    @property
    def std(self):
        """
        Returns the current running standard deviation.
        """
        # Variance = (Sum of Squares) / Count - (Mean)^2
        # Since we're not subtracting the mean for normalization,
        # we effectively treat the mean as 0 for this scaling.
        # So, variance approx = Sum of Squares / Count

        # Avoid division by zero when count is 0
        if self.count == 0:
            return torch.ones(self.state_dim, device=self.device), torch.ones(self.action_dim, device=self.device) # Return 1s if no data yet

        current_variance_states = self.sum_sq_states / (self.count + self.epsilon)
        current_variance_actions = self.sum_sq_actions / (self.count + self.epsilon)

        std_states = torch.sqrt(current_variance_states + self.epsilon).to(self.device)
        std_actions = torch.sqrt(current_variance_actions + self.epsilon).to(self.device)

        # Set std to 1 where it is 0 to account for constant values in the obs
        std_states[std_states == 0] = 1.0
        std_actions[std_actions == 0] = 1.0

        return std_states, std_actions

    def normalize_states(self, states):
        """
        Normalizes the states by dividing by the current running standard deviation.

        Args:
            states (torch.Tensor): A batch of states to normalize.
                                   Expected shape (batch_size, state_dim).

        Returns:
            torch.Tensor: The normalized states.
        """
        return states / self.std[0]

    def normalize(self, states, actions = None):
        """
        Normalizes the input observations by dividing by the current running standard deviation.

        Args:
            states (torch.Tensor): A batch of states to normalize.
                                   Expected shape (batch_size, state_dim).
            actions (torch.Tensor): A batch of actions to normalize.
                                Expected shape (batch_size, action_dim).

        Returns:
            torch.Tensor: The normalized observations.
        """
        if actions is None:
            return states / self.std[0]
        else:
            return states / self.std[0], actions / self.std[1]

    def denormalize(self, normalized_states, normalized_actions):
        """
        Denormalizes previously normalized states and actions.

        Args:
            normalized_states (torch.Tensor): A batch of normalized states.
            normalized_actions (torch.Tensor): A batch of normalized actions.

        Returns:
            torch.Tensor: The original (denormalized) states and actions.
        """
        return normalized_states * self.std[0], normalized_actions * self.std[1]

    def state_dict(self):
        """Returns a dictionary containing the current state of the normalizer."""
        return {
            'sum_sq_states': self.sum_sq_states,
            'sum_sq_actions': self.sum_sq_actions,
            'count': self.count
        }

    def load_state_dict(self, state_dict):
        """Loads the state of the normalizer from a state_dict."""
        self.sum_sq_states = state_dict['sum_sq_states'].to(self.device)
        self.sum_sq_actions = state_dict['sum_sq_actions'].to(self.device)
        self.count = state_dict['count']