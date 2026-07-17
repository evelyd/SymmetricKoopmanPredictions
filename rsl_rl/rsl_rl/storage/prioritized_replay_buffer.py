import torch
import numpy as np
from collections import deque
from math import floor
from dha.data.DhaDynamicsRecording import DhaDynamicsRecording

# This stores the priorities in a binary tree structure
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # tree stores priorities, leaves are at capacity-1 to 2*capacity-2
        self.tree = np.zeros(2 * capacity - 1)
        # data stores actual indices to the experience buffer
        self.data_indices = np.zeros(capacity, dtype=int)
        self.data_pointer = 0 # Points to the next available slot in data_indices
        self.n_entries = 0 # Number of actual entries in the buffer

    def add(self, priority, data_idx):
        # Calculate leaf index in the tree array
        tree_idx = self.data_pointer + self.capacity - 1
        self.data_indices[self.data_pointer] = data_idx
        self.update(tree_idx, priority) # Update priority and propagate up the tree

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        # Retrieve the leaf index, priority, and data_idx for a given cumulative sum 's'
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree): # Reached a leaf node
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = self.data_indices[leaf_idx - self.capacity + 1] # Get original data index
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0] # Root of the tree contains the total sum of priorities

    def __len__(self):
        return self.n_entries

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples with prioritization."""

    def __init__(self, obs_dim, action_dim, beta_initial, beta_annealing_steps, buffer_size, device="cpu", alpha=0.6):
        """Initialize a PrioritizedReplayBuffer object.
        Arguments:
            obs_dim (int): Dimension of observations.
            action_dim (int): Dimension of actions.
            buffer_size (int): maximum size of buffer
            device (str): Device to store tensors ('cpu' or 'cuda').
            alpha (float): Priority exponent (0 = uniform, 1 = full prioritization).
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.actions = torch.zeros(buffer_size, action_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device
        self.beta_initial = beta_initial
        self.beta_annealing_steps = beta_annealing_steps

        self.alpha = alpha  # Priority exponent
        self.epsilon = 1e-6 # Small constant to ensure non-zero priority
        self.max_priority = 1.0 # Initial max priority for new samples

        self.tree = SumTree(buffer_size)

        self.step = 0 # This now tracks the position in the circular buffer for states/actions/next_states
        self.num_samples = 0 # This tracks the actual number of elements currently in the buffer

    def __len__(self):
        return self.num_samples

    def insert(self, states, actions, next_states):
        """Add new states to memory.
        Arguments:
            states (torch.Tensor): A batch of states. Shape (batch_size, obs_dim)
            actions (torch.Tensor): A batch of actions. Shape (batch_size, action_dim)
            next_states (torch.Tensor): A batch of next states. Shape (batch_size, obs_dim)
        """
        num_new_samples = states.shape[0]

        # Use the current max_priority for new samples
        # This gives new samples a high probability of being sampled at least once
        initial_priority = self.max_priority

        for i in range(num_new_samples):
            # Store data in the circular buffer
            self.states[self.step] = states[i]
            self.actions[self.step] = actions[i]
            self.next_states[self.step] = next_states[i]

            # Add to SumTree with initial priority, storing the actual index in the data buffer
            self.tree.add(initial_priority, self.step)

            self.step = (self.step + 1) % self.buffer_size
            self.num_samples = min(self.buffer_size, self.num_samples + 1)

    def sample(self, mini_batch_size, beta):
        """Sample a batch of experiences with priorities."""
        batch_indices_in_tree = []  # Indices within the SumTree (leaf nodes)
        batch_data_indices = []     # Actual indices in the states/actions/next_states tensors
        batch_priorities = []       # Priorities of the sampled experiences
        batch_data = []             # The (s, a, s') tuples

        segment = self.tree.total_priority / mini_batch_size

        for i in range(mini_batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b) # Sample a value uniformly within each segment

            tree_idx, priority, data_idx = self.tree.get(s)

            batch_indices_in_tree.append(tree_idx)
            batch_data_indices.append(data_idx)
            batch_priorities.append(priority)

        # Retrieve data from the main tensors using collected data_indices
        states = self.states[batch_data_indices].to(self.device)
        actions = self.actions[batch_data_indices].to(self.device)
        next_states = self.next_states[batch_data_indices].to(self.device)

        # Compute Importance Sampling weights
        # P(i) = priority_i / total_priority
        # is_weight = (N * P(i))^(-beta) / max(is_weight)

        # Avoid division by zero if total_priority is 0 (shouldn't happen if samples are added)
        if self.tree.total_priority == 0:
            is_weights = torch.ones(mini_batch_size, dtype=torch.float32, device=self.device)
        else:
            sampling_probabilities = torch.tensor(batch_priorities, dtype=torch.float32, device=self.device) / self.tree.total_priority
            is_weights = (self.num_samples * sampling_probabilities)**(-beta)
            # Normalize IS weights by their maximum value to stabilize training
            is_weights /= torch.max(is_weights)

        return states, actions, next_states, batch_indices_in_tree, is_weights

    def update_priorities(self, tree_indices, errors):
        """Update priorities in the SumTree based on new errors."""
        for tree_idx, error in zip(tree_indices, errors):
            # The error is typically the TD error or a loss
            # priority = (|error| + self.epsilon)*self.alpha
            priority = (abs(error) + self.epsilon)**self.alpha
            self.tree.update(tree_idx, priority)
            # Keep track of the maximum priority seen
            self.max_priority = max(self.max_priority, priority)

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
        """Generator that yields observation samples of length `n_frames_per_state` from the Markov Dynamics recordings."""
        traj_length = states.shape[0]
        steps_in_pred_horizon = prediction_horizon
        assert steps_in_pred_horizon > 0, f"Invalid prediction horizon {steps_in_pred_horizon}"

        # Ensure inputs are on CPU for numpy conversion
        states_np = states.cpu().numpy()
        actions_np = actions.cpu().numpy()
        next_states_np = next_states.cpu().numpy()

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
            obs_time_horizon = states_np[start_indices[0] : end_indices[-1]]
            next_obs_time_horizon = next_states_np[start_indices[0] : end_indices[-1]]
            action_time_horizon = actions_np[start_indices[0] : end_indices[-1]]
            # Reshape the slice to have the desired shape (time, frames_per_step, obs_dim)
            obs_time_horizon = obs_time_horizon.reshape((num_steps, frames_per_step, states_np.shape[1]))
            next_obs_time_horizon = next_obs_time_horizon.reshape((num_steps, frames_per_step, next_states_np.shape[1]))
            action_time_horizon = action_time_horizon.reshape((num_steps, frames_per_step, actions_np.shape[1]))

            sample["state_observations"] = torch.from_numpy(obs_time_horizon).to(states.device)
            sample["action_observations"] = torch.from_numpy(action_time_horizon).to(actions.device)
            sample["next_state_observations"] = torch.from_numpy(next_obs_time_horizon).to(next_states.device)
            yield sample