import logging
import math
from typing import Optional, Union, Dict

import torch
from morpho_symm.nn.MLP import MLP
from torch import Tensor
from plotly.graph_objs import Figure

from dha.nn.DynamicsAutoEncoder import DAE
from dha.nn.ControlledLinearDynamics import ControlledLinearDynamics
from dha.nn.markov_dynamics import MarkovDynamics
from dha.utils.losses_and_metrics import obs_state_space_metrics
from dha.utils.mysc import traj_from_states

log = logging.getLogger(__name__)


class ControlledDAE(DAE):
    _default_obs_fn_params = dict(
        num_layers=4,
        num_hidden_units=128,
        activation=torch.nn.ELU,
        batch_norm=True,
        bias=False,
        init_mode="fan_in",
    )

    def __init__(
        self,
        action_dim: int,
        transfer_op_eigval_reg_w: float = 100.0,
        **dae_params,
    ):
        self.action_dim = action_dim
        self.transfer_op_eigval_reg_w = transfer_op_eigval_reg_w

        # Pass the obs_state_dynamics to the parent class initializer
        super(ControlledDAE, self).__init__(**dae_params)

    def forward(self, state: Tensor, action: Tensor, next_state: Optional[Tensor]) -> [Dict[str, Tensor]]:
        """Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.

        Args:
            state: (batch, state_dim) Initial state of the system.
            action: (batch, action_dim) Action applied to the system.
            next_state: (batch, pred_horizon, state_dim) Next states of the system in a prediction horizon window.

        Returns:
            predictions (dict): A dictionary containing the predicted state and observable state trajectory.
                - 'obs_state_traj': (batch, pred_horizon + 1, obs_state_dim)
                - 'pred_obs_state_traj': (batch, pred_horizon + 1, obs_state_dim)
                - 'pred_state_traj': (batch, pred_horizon + 1, state_dim)

        """
        assert state.shape[-1] == next_state.shape[-1], f"Invalid state dimension {state.shape[-1]} != {self.state_dim}"
        assert state.shape[0] == next_state.shape[0], f"Invalid batch size {state.shape[0]} != {next_state.shape[0]}"
        assert len(state.shape) == 2, f"Invalid state shape {state.shape}. Expected (batch, {self.state_dim})"

        assert action.shape[0] == action.shape[0], f"Invalid action batch size {action.shape[0]} != {action.shape[0]}"
        assert action.shape[-1] == self.action_dim, f"Invalid action dimension {action.shape[-1]} != {self.action_dim}"
        if len(next_state.shape) == 2:
            next_state = next_state.unsqueeze(1)

        batch, pred_horizon, _ = next_state.shape
        time_horizon = pred_horizon + 1

        assert len(action.shape) == 3, f"Invalid action shape {action.shape}. Expected (batch, {pred_horizon}, {self.action_dim})"

        # Apply pre-processing to the initial state and state trajectory
        # obtaining a stare trajectory of shape: (batch * (pred_horizon + 1), state_dim) tensor
        state_traj = self.pre_process_state(state=state, next_state=next_state)

        # Preprocess the action trajectory
        action_traj = self.pre_process_state(state=action)

        # Observation function evaluation ===============================================
        # Compute the projection of the state trajectory in the main and auxiliary observable states
        obs_fn_output = self.obs_fn(state_traj)
        # Post-process observation state trajectories to get (batch, (pred_horizon + 1), obs_state_dim) tensors
        if not isinstance(obs_fn_output, tuple):
            pre_obs_fn_output = self.pre_process_obs_state(obs_fn_output)
        else:
            pre_obs_fn_output = self.pre_process_obs_state(*obs_fn_output)

        # Extract the observable state trajectory
        assert "obs_state_traj" in pre_obs_fn_output, f"Missing 'obs_state_traj' in {pre_obs_fn_output}"
        obs_state_traj = pre_obs_fn_output.pop("obs_state_traj")

        # Evolution of observable states ===============================================
        # Evolve the observable state with the current observable dynamics model.
        obs_dyn_output = self.obs_space_dynamics(
            state=obs_state_traj[:, 0, :], action=action_traj, next_state=obs_state_traj[:, 1:, :], **pre_obs_fn_output
        )
        obs_dyn_output = {k.replace("state", "obs_state"): v for k, v in obs_dyn_output.items()}
        pred_obs_state_traj = obs_dyn_output.pop("pred_obs_state_traj")

        # Observation function inversion ===============================================
        # This post-processing of observables ensures the input to the inverse function is of the correct shape.
        # Predicted trajectory of observable state in observable space
        post_pred_obs_dyn_output = self.post_process_obs_state(pred_obs_state_traj)
        # Predicted trajectory of the system's state in the original state space
        pred_state_traj = self.inv_obs_fn(post_pred_obs_dyn_output.pop("obs_state_traj"))
        # Ground-truth trajectory of observable state in observable space
        post_obs_dyn_output = self.post_process_obs_state(obs_state_traj)
        # Reconstruction of the system's state in the original state space
        rec_state_traj = self.inv_obs_fn(post_obs_dyn_output.pop("obs_state_traj"))

        # Apply the required post-processing of the state.
        pred_state_traj = self.post_process_state(pred_state_traj)
        rec_state_traj = self.post_process_state(rec_state_traj)

        # Sanity checks of shapes.
        self.check_state_traj_shape(
            pred_state_traj=pred_state_traj,
            rec_state_traj=rec_state_traj,
            time_horizon=time_horizon,
            state_dim=self.state_dim,
        )
        self.check_state_traj_shape(
            obs_state_traj=obs_state_traj,
            pred_obs_state_traj=pred_obs_state_traj,
            time_horizon=time_horizon,
            state_dim=self.obs_state_dim,
        )
        return dict(
            obs_state_traj=obs_state_traj,
            pred_obs_state_traj=pred_obs_state_traj,
            pred_state_traj=pred_state_traj,
            rec_state_traj=rec_state_traj,
            **pre_obs_fn_output,
            **obs_dyn_output,
        )

    def forecast(self, state: Tensor, action: Tensor, n_steps: int = 1, **kwargs) -> [Dict[str, Tensor]]:
        """Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.

        This function uses the empirical transfer operator to compute forcast the observable state.

        Args:
            state: (batch_dim, obs_state_dim) Initial observable state of the system.
            action: (batch_dim, action_dim) Action applied to the system.
            n_steps: Number of steps to predict.
            **kwargs:
        Returns:
            pred_next_obs_state: (batch_dim, n_steps, obs_state_dim) Predicted observable state.

        """
        assert state.shape[-1] == self.state_dim, f"Invalid state: {state.shape}. Expected (batch, {self.state_dim})"
        assert action.shape[-1] == self.action_dim, f"Invalid action: {action.shape}. Expected (batch, {self.action_dim})"
        time_horizon = n_steps + 1

        obs_state = self.obs_fn(state)

        pred_obs_state_traj = self.obs_space_dynamics.forcast(state=obs_state, action=action, n_steps=n_steps)

        pred_state_traj = self.inv_obs_fn(pred_obs_state_traj)

        if self._batch_size is None:
            self._batch_size = state.shape[0]

        assert pred_state_traj.shape[-2:] == (time_horizon, self.state_dim), (
            f"{pred_state_traj.shape[-2:]}!=({time_horizon}, {self.state_dim})"
        )
        assert pred_obs_state_traj.shape[-2:] == (time_horizon, self.obs_state_dim), (
            f"{pred_obs_state_traj.shape[-2:]}!=({time_horizon}, {self.obs_state_dim})"
        )
        return pred_state_traj, pred_obs_state_traj

    def compute_loss_and_metrics(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        pred_state_traj: Tensor,
        rec_state_traj: Tensor,
        obs_state_traj: Tensor,
        pred_obs_state_traj: Tensor,
        pred_obs_state_one_step: Tensor,
    ) -> (Tensor, Dict[str, Tensor]):
        _, forecast_metrics = super(DAE, self).compute_loss_and_metrics(
            state=state,
            next_state=next_state,
            pred_state_traj=pred_state_traj,
            rec_state_traj=rec_state_traj,
            obs_state_traj=obs_state_traj,
            pred_obs_state_traj=pred_obs_state_traj,
        )

        # obs_space_metrics = self.get_obs_space_metrics(obs_state_traj, pred_obs_state_one_step)

        # Add the eigenvalue penalty to the loss
        loss = self.compute_loss(
            state_rec_loss=forecast_metrics["state_rec_loss"],
            state_pred_loss=forecast_metrics["state_pred_loss"],
            obs_pred_loss=forecast_metrics["obs_pred_loss"],
        )

        # metrics = dict(**forecast_metrics, **obs_space_metrics)
        metrics = dict(**forecast_metrics)
        return loss, metrics

    def compute_loss(
        self, state_rec_loss: Tensor, state_pred_loss: Tensor, obs_pred_loss: Tensor, orth_reg: Optional[Tensor] = None
    ):

        # Compute the autoencoder loss, which is a combination of state reconstruction, state prediction, and observation prediction losses
        loss = super(ControlledDAE, self).compute_loss(
            state_rec_loss=state_rec_loss,
            state_pred_loss=state_pred_loss,
            obs_pred_loss=obs_pred_loss,
            orth_reg=orth_reg,
        )

        return loss

    @torch.no_grad()
    def eval_metrics(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        obs_state_traj: Tensor,
        obs_state_traj_aux: Optional[Tensor] = None,
        pred_state_traj: Optional[Tensor] = None,
        rec_state_traj: Optional[Tensor] = None,
        pred_obs_state_one_step: Optional[Tensor] = None,
        pred_obs_state_traj: Optional[Tensor] = None,
    ) -> (Dict[str, Figure], Dict[str, Tensor]):
        state_traj = traj_from_states(state, next_state)

        if obs_state_traj_aux is None and pred_obs_state_one_step is not None:
            obs_state_traj_aux = pred_obs_state_one_step

        # Detach all arguments and ensure they are in CPU
        state_traj = state_traj.detach().cpu().numpy()
        obs_state_traj = obs_state_traj.detach().cpu().numpy()
        if obs_state_traj_aux is not None:
            obs_state_traj_aux = obs_state_traj_aux.detach().cpu().numpy()
        if pred_state_traj is not None:
            pred_state_traj = pred_state_traj.detach().cpu().numpy()
        if pred_obs_state_traj is not None:
            pred_obs_state_traj = pred_obs_state_traj.detach().cpu().numpy()

        from dha.utils.plotting import plot_two_panel_trajectories
        fig = plot_two_panel_trajectories(
            state_trajs=state_traj,
            pred_state_trajs=pred_state_traj,
            obs_state_trajs=obs_state_traj,
            pred_obs_state_trajs=pred_obs_state_traj,
            dt=self.dt,
            n_trajs_to_show=5,
        )
        figs = dict(prediction=fig)
        if self.obs_state_dim == 3:
            from dha.utils.plotting import plot_system_3D
            fig_3do = plot_system_3D(
                trajectories=obs_state_traj,
                secondary_trajectories=pred_obs_state_traj,
                title="obs_state",
                num_trajs_to_show=20,
            )
            if obs_state_traj_aux is not None:
                fig_3do = plot_system_3D(
                    trajectories=obs_state_traj_aux,
                    legendgroup="aux",
                    traj_colorscale="solar",
                    num_trajs_to_show=20,
                    fig=fig_3do,
                )
            figs["obs_state"] = fig_3do
        if self.state_dim == 3:
            fig_3ds = plot_system_3D(
                trajectories=state_traj,
                secondary_trajectories=pred_state_traj,
                title="state_traj",
                num_trajs_to_show=20,
            )
            figs["state"] = fig_3ds

        if self.obs_state_dim == 2:
            from dha.utils.plotting import plot_system_2D
            fig_2do = plot_system_2D(
                trajs=obs_state_traj, secondary_trajs=pred_obs_state_traj, alpha=0.2, num_trajs_to_show=10
            )
            if obs_state_traj_aux is not None:
                fig_2do = plot_system_2D(trajs=obs_state_traj_aux, legendgroup="aux", num_trajs_to_show=10, fig=fig_2do)
            figs["obs_state"] = fig_2do
        if self.state_dim == 2:
            fig_2ds = plot_system_2D(trajs=state_traj, secondary_trajs=pred_state_traj, alpha=0.2, num_trajs_to_show=10)
            figs["state"] = fig_2ds

        metrics = None
        return figs, metrics

    def build_obs_dyn_module(self) -> ControlledLinearDynamics:
        return ControlledLinearDynamics(state_dim=self.obs_state_dim, action_dim=self.action_dim, dt=self.dt, trainable=True, bias=self.enforce_constant_fn)
