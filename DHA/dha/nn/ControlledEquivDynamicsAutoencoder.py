import logging
import math
from typing import Optional, Union, Dict, Tuple

import escnn
import torch
from escnn.group import Representation
from escnn.nn import FieldType, GeometricTensor # Figure from plotly.graph_objs
from plotly.graph_objs import Figure
from torch import Tensor

from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.DynamicsAutoEncoder import DAE
from dha.nn.ControlledEquivLinearDynamics import ControlledEquivLinearDynamics
from dha.nn.markov_dynamics import MarkovDynamics
from dha.utils.mysc import batched_to_flat_trajectory, traj_from_states

log = logging.getLogger(__name__)


class ControlledEquivDAE(EquivDAE):
    _default_obs_fn_params = dict(
        num_layers=4,
        num_hidden_units=128,  # Approximate number of neurons in hidden layers. Actual number depends on group order.
        activation="p_elu",
        batch_norm=True,
        bias=False,
    )

    def __init__(
        self,
        state_rep: Representation,
        action_rep: Representation,
        obs_state_dim: int,
        dt: Union[float, int] = 1,
        obs_fn_params: Optional[dict] = None,
        group_avg_trick: bool = True,
        state_dependent_obs_dyn: bool = False,
        # DAE specific params (passed via dae_kwargs to EquivDAE -> DAE)
        obs_pred_w: float = 0.1,
        orth_w: float = 0.1,
        corr_w: float = 0.0, # DAE param, not in EquivDAE explicitly
        enforce_constant_fn: bool = True, # DAE param
        # Other markov_dyn_params for DAE -> LatentMarkovDynamics -> MarkovDynamics
        **markov_dyn_params,
    ):
        self.symm_group = state_rep.group
        self.gspace = escnn.gspaces.no_base_space(self.symm_group)
        self.action_rep = action_rep
        self.action_type = FieldType(self.gspace, [action_rep])
        self.action_dim = self.action_type.size

        super().__init__(
            state_rep=state_rep,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_fn_params=obs_fn_params,
            group_avg_trick=group_avg_trick,
            state_dependent_obs_dyn=state_dependent_obs_dyn,
            obs_pred_w=obs_pred_w,
            orth_w=orth_w,
            corr_w=corr_w,
            enforce_constant_fn=enforce_constant_fn,
            **markov_dyn_params
        )

    def build_obs_dyn_module(self) -> ControlledEquivLinearDynamics:
        return ControlledEquivLinearDynamics(
            state_type=self.obs_state_type,
            action_type=self.action_type,
            dt=self.dt,
            trainable=True,
            bias=self.enforce_constant_fn,
        )

    def pre_process_action(self, action: Tensor) -> GeometricTensor:
        """
        Input action: (batch, pred_horizon, action_dim_tensor)
        Output: GeometricTensor (batch * pred_horizon, action_dim_geom)
        """
        action_flat_tensor = action.reshape(-1, self.action_dim)
        action_gt_flat = self.action_type(action_flat_tensor)
        return action_gt_flat

    def forward(self, state: Tensor, action: Tensor, next_state: Optional[Tensor]) -> Dict[str, Tensor]:
        assert state.shape[-1] == self.state_dim, f"Invalid state dimension {state.shape[-1]} != {self.state_dim}"
        assert state.shape[0] == next_state.shape[0], f"Invalid batch size {state.shape[0]} != {next_state.shape[0]}"
        assert len(state.shape) == 2, f"Invalid state shape {state.shape}. Expected (batch, {self.state_dim})"

        if len(next_state.shape) == 2:
            next_state = next_state.unsqueeze(1)

        batch_size, pred_horizon, _ = next_state.shape
        time_horizon = pred_horizon + 1

        assert action.shape[0] == batch_size, f"Invalid action batch size {action.shape[0]} != {batch_size}"
        assert action.shape[-1] == self.action_dim, f"Invalid action dimension {action.shape[-1]} != {self.action_dim}"
        assert len(action.shape) == 3, f"Invalid action shape {action.shape}. Expected (batch, {pred_horizon}, {self.action_dim})"

        state_traj_gt: GeometricTensor = self.pre_process_state(state=state, next_state=next_state)
        action_traj_processed: GeometricTensor = self.pre_process_action(action)

        obs_fn_output: GeometricTensor = self.obs_fn(state_traj_gt)
        pre_obs_fn_output: dict[str, Tensor] = self.pre_process_obs_state(obs_fn_output) # Handles tuple output from obs_fn
        obs_state_traj: Tensor = pre_obs_fn_output.pop("obs_state_traj")

        initial_obs_state_geom: GeometricTensor = self.obs_state_type(obs_state_traj[:, 0, :])

        # Prepare next_state for dynamics module if it's used for teacher forcing/loss calculation inside dynamics
        next_obs_state_flat_gt = None
        if obs_state_traj.shape[1] > 1: # if pred_horizon > 0
            next_obs_state_flat_gt = self.obs_state_type(batched_to_flat_trajectory(obs_state_traj[:, 1:, :]))

        obs_dyn_output: dict[str, Tensor] = self.obs_space_dynamics(
            state=initial_obs_state_geom,
            action=action_traj_processed,
            next_state=next_obs_state_flat_gt, # Pass flat GT, dynamics should handle unflattening if needed
            **pre_obs_fn_output
        )
        obs_dyn_output = {k.replace("state", "obs_state"): v for k, v in obs_dyn_output.items()}
        pred_obs_state_traj: Tensor = obs_dyn_output.pop("pred_obs_state_traj")

        post_pred_obs_dyn_output: dict[str, GeometricTensor] = self.post_process_obs_state(pred_obs_state_traj)
        pred_state_traj_geom: GeometricTensor = self.inv_obs_fn(post_pred_obs_dyn_output.pop("obs_state_traj"))
        pred_state_traj: Tensor = self.post_process_state(pred_state_traj_geom)

        post_obs_dyn_output: dict[str, GeometricTensor] = self.post_process_obs_state(obs_state_traj)
        rec_state_traj_geom: GeometricTensor = self.inv_obs_fn(post_obs_dyn_output.pop("obs_state_traj"))
        rec_state_traj: Tensor = self.post_process_state(rec_state_traj_geom)

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

    def forecast(self, state: Tensor, action: Tensor, n_steps: int = 1, **kwargs) -> Tuple[Tensor, Tensor]:
        assert state.shape[-1] == self.state_dim, f"Invalid state: {state.shape}. Expected (batch, {self.state_dim})"
        assert action.shape[-1] == self.action_dim, f"Invalid action: {action.shape}. Expected (batch, {n_steps}, {self.action_dim})"
        assert action.shape[1] == n_steps, f"Action sequence length {action.shape[1]} != n_steps {n_steps}"

        time_horizon = n_steps + 1
        batch_size = state.shape[0]

        initial_state_gt: GeometricTensor = self.pre_process_state(state=state)
        initial_obs_state_gt: GeometricTensor = self.obs_fn(initial_state_gt)
        action_sequence_gt_flat: GeometricTensor = self.pre_process_action(action)

        # obs_space_dynamics.forcast returns Tensor (batch, n_steps + 1, obs_dim_tensor)
        pred_obs_state_traj_tensor: Tensor = self.obs_space_dynamics.forcast(
            state=initial_obs_state_gt,
            action=action_sequence_gt_flat,
            n_steps=n_steps
        )

        flat_pred_obs_state_traj_tensor = batched_to_flat_trajectory(pred_obs_state_traj_tensor)
        flat_pred_obs_state_traj_gt = self.obs_state_type(flat_pred_obs_state_traj_tensor)

        pred_state_traj_gt: GeometricTensor = self.inv_obs_fn(flat_pred_obs_state_traj_gt)
        pred_state_traj: Tensor = self.post_process_state(pred_state_traj_gt)

        if self._batch_size is None: self._batch_size = batch_size

        assert pred_state_traj.shape == (batch_size, time_horizon, self.state_dim), \
            f"{pred_state_traj.shape}!=({batch_size}, {time_horizon}, {self.state_dim})"
        assert pred_obs_state_traj_tensor.shape == (batch_size, time_horizon, self.obs_state_dim), \
            f"{pred_obs_state_traj_tensor.shape}!=({batch_size}, {time_horizon}, {self.obs_state_dim})"

        return pred_state_traj, pred_obs_state_traj_tensor

    @torch.no_grad()
    def eval_metrics(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        obs_state_traj: Tensor,
        pred_obs_state_one_step: Optional[Tensor] = None,
        pred_state_traj: Optional[Tensor] = None,
        rec_state_traj: Optional[Tensor] = None,
        pred_obs_state_traj: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Figure], Dict[str, Tensor]]:

        state_traj_np = traj_from_states(state, next_state).detach().cpu().numpy()
        obs_state_traj_np = obs_state_traj.detach().cpu().numpy()

        obs_state_traj_aux_np = None
        if pred_obs_state_one_step is not None:
             obs_state_traj_aux_np = pred_obs_state_one_step.detach().cpu().numpy()

        pred_state_traj_np = pred_state_traj.detach().cpu().numpy() if pred_state_traj is not None else None
        pred_obs_state_traj_np = pred_obs_state_traj.detach().cpu().numpy() if pred_obs_state_traj is not None else None

        from dha.utils.plotting import plot_two_panel_trajectories
        fig = plot_two_panel_trajectories(
            state_trajs=state_traj_np,
            pred_state_trajs=pred_state_traj_np,
            obs_state_trajs=obs_state_traj_np,
            pred_obs_state_trajs=pred_obs_state_traj_np,
            dt=self.dt,
            n_trajs_to_show=5,
        )
        figs = dict(prediction=fig)

        if self.obs_state_dim == 3:
            from dha.utils.plotting import plot_system_3D
            fig_3do = plot_system_3D(
                trajectories=obs_state_traj_np,
                secondary_trajectories=pred_obs_state_traj_np,
                title="obs_state",
                num_trajs_to_show=20,
            )
            if obs_state_traj_aux_np is not None:
                fig_3do = plot_system_3D(
                    trajectories=obs_state_traj_aux_np,
                    legendgroup="aux",
                    traj_colorscale="solar",
                    num_trajs_to_show=20,
                    fig=fig_3do,
                )
            figs["obs_state"] = fig_3do
        if self.state_dim == 3:
            from dha.utils.plotting import plot_system_3D
            fig_3ds = plot_system_3D(
                trajectories=state_traj_np,
                secondary_trajectories=pred_state_traj_np,
                title="state_traj",
                num_trajs_to_show=20,
            )
            figs["state"] = fig_3ds

        if self.obs_state_dim == 2:
            from dha.utils.plotting import plot_system_2D
            fig_2do = plot_system_2D(
                trajs=obs_state_traj_np, secondary_trajs=pred_obs_state_traj_np, alpha=0.2, num_trajs_to_show=10
            )
            if obs_state_traj_aux_np is not None:
                fig_2do = plot_system_2D(trajs=obs_state_traj_aux_np, legendgroup="aux", num_trajs_to_show=10, fig=fig_2do)
            figs["obs_state"] = fig_2do
        if self.state_dim == 2:
            from dha.utils.plotting import plot_system_2D
            fig_2ds = plot_system_2D(trajs=state_traj_np, secondary_trajs=pred_state_traj_np, alpha=0.2, num_trajs_to_show=10)
            figs["state"] = fig_2ds

        metrics = None
        return figs, metrics

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
