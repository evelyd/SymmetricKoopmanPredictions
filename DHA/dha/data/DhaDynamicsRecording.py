from morpho_symm.data.DynamicsRecording import DynamicsRecording
import numpy as np
from typing import Iterable, List, Optional, Union
from dha.utils.mysc import safe_standardize

class DhaDynamicsRecording(DynamicsRecording):
    """
    DhaDynamicsRecording is a subclass of DynamicsRecording that provides additional functionality for handling action data.
    """
    def __post_init__(self):
        super().__post_init__()

    def action_moments(self) -> [np.ndarray, np.ndarray]:
        """Compute the mean and standard deviation of the action observations."""
        mean, var = [], []
        for obs_name in self.action_obs:
            if obs_name not in self.obs_moments.keys():
                self.compute_obs_moments(obs_name)
            obs_mean, obs_var = self.obs_moments[obs_name]
            mean.append(obs_mean)
            var.append(obs_var)
        mean, var = np.concatenate(mean), np.concatenate(var)
        return mean, var

    @staticmethod
    def map_action_nsteps_action(
        sample: dict,
        action_observations: List[str],
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
    ) -> dict:
        """Map composing multiple frames of observations into a flat vectors `action` and `next_action` samples.

        This method constructs the action `a_t` and history of nex steps `a_t+1` of the Markov Process.
        The action is defined as a set of observations within a window of fps=`frames_per_action`.
        E.g.: Consider the action is defined by the observations [m=momentum, p=position] at `fps` consecutive frames.
            Then the action at time `t` is defined as `a_t = [m_f, m_f+1,..., m_f+fps, p_f, p_f+1, ..., p_f+fps]`.
            Where we use f to denote frame in time to make the distinction from the time index `t` of the Markov
            Process.
            Then, the next action is defined as `a_t+1 = [m_f+fps,..., m_fps+fps, p_f+fps, ..., p_f+fps+fps]`.

        Args:
            sample (dict): Dictionary containing the observations of the system of shape [action_time, f].
            action_observations: Ordered list of observations names composing the action space.

        Returns:
            A dictionary containing the MDP action and the next_action/s `[a_t, a_t+1, a_t+2, ..., a_t+pred_horizon]`.
        """
        batch_size = len(sample[f"{action_observations[0]}"])
        time_horizon = len(sample[f"{action_observations[0]}"][0])
        # Flatten observations a_t = [a_f, a_f+1, af+2, ..., a_f+F] s.t. a_t in R^{F * dim(a)}, a_f in R^{dim(a)}
        action_obs = [sample[m] for m in action_observations]
        # Define the action at time t and the actions at time [t+1, t+pred_horizon]
        action_traj = np.concatenate(action_obs, axis=-1).reshape(batch_size, time_horizon, -1)
        if action_mean is not None and action_std is not None:
            action_traj = safe_standardize(action_traj, action_mean, action_std)
        return dict(action=action_traj[:, :-1])

    @staticmethod
    def map_state_action_state(
        sample: dict,
        state_observations: List[str],
        action_observations: List[str],
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        action_mean: Optional[np.ndarray] = None, # Add action normalization params
        action_std: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Map composing multiple observations to state, next_state, and current action samples.
        This function discards the 'next_action' component.
        """

        state_temp_dict = DynamicsRecording.map_state_next_state(
            sample,
            state_observations,
            state_mean=state_mean,
            state_std=state_std,
        )
        flat_sample = {
            "state": state_temp_dict["state"],
            "next_state": state_temp_dict["next_state"],
        }

        action_temp_dict = DhaDynamicsRecording.map_action_nsteps_action(
            sample,
            action_observations,
            action_mean=action_mean, # Apply action-specific normalization
            action_std=action_std,
        )
        flat_sample["action"] = action_temp_dict["action"]

        return flat_sample