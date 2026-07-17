import logging
from typing import Optional, Union, Dict

import torch
import escnn
from escnn.group import Representation
from escnn.nn import FieldType, GeometricTensor
from torch import Tensor

from dha.nn.EquivLinearDynamics import EquivLinearDynamics
from dha.utils.mysc import batched_to_flat_trajectory, traj_from_states

log = logging.getLogger(__name__)

class ControlledEquivLinearDynamics(EquivLinearDynamics):
    def __init__(
        self,
        state_type: FieldType,
        action_type: FieldType,
        dt: Optional[Union[float, int]] = 1,
        trainable=False,
        bias: bool = True,
        init_mode: str = "identity",
        **markov_dyn_kwargs,
    ):
        self.action_type = action_type
        self.action_dim = action_type.size

        super().__init__(
            state_type=state_type,
            dt=dt,
            trainable=trainable,
            bias=bias,
            init_mode=init_mode, # For transfer_op (A)
            **markov_dyn_kwargs
        )

        if not trainable:
            self.control_op_matrix = None
            self.control_op_bias_vector = None
        else:
            self.control_op = self.build_control_linear_map()
            self.reset_control_parameters(init_mode="stable") # Or a specific init for control_op (B)

    def build_control_linear_map(self) -> escnn.nn.Linear:
        """ Builds the equivariant linear map for the control input B: U -> X """
        return escnn.nn.Linear(
            in_type=self.action_type,
            out_type=self.state_type,
            bias=self.bias, # Consistent with transfer_op bias
            initialize=False # Manual initialization via reset_control_parameters
        )

    def reset_control_parameters(self, init_mode: str):
        if init_mode == "zero":
            if hasattr(self.control_op, 'weights') and self.control_op.weights is not None:
                 self.control_op.weights.data.zero_()
            if self.bias and hasattr(self.control_op, 'bias') and self.control_op.bias is not None:
                self.control_op.bias.data.zero_()
            log.info(f"Equivariant control operator initialized to zero.")
        elif init_mode == "stable": # Defaulting to zero for "stable" as identity is tricky
            if hasattr(self.control_op, 'weights') and self.control_op.weights is not None:
                 self.control_op.weights.data.zero_()
            if self.bias and hasattr(self.control_op, 'bias') and self.control_op.bias is not None:
                self.control_op.bias.data.zero_()
            log.info(f"Equivariant control operator initialized to zero (as 'stable' init).")
        else:
            log.warning(f"Control op init mode '{init_mode}' not fully implemented for equivariant; using small random weights.")
            if hasattr(self.control_op, 'weights') and self.control_op.weights is not None:
                 self.control_op.weights.data.normal_(0, 0.01)
            if self.bias and hasattr(self.control_op, 'bias') and self.control_op.bias is not None:
                self.control_op.bias.data.zero_()

    def forward(self, state: GeometricTensor, action: GeometricTensor,
                next_state: Optional[GeometricTensor] = None, **kwargs) -> Dict[str, Tensor]:
        """
        Evolves the state over a prediction horizon using actions.
        Args:
            state: Initial state as GeometricTensor (batch, state_dim_geom).
            action: Action sequence as GeometricTensor (batch * pred_horizon, action_dim_geom).
            next_state: Ground truth next states as GeometricTensor (batch * pred_horizon, state_dim_geom), used to infer pred_horizon.
        Returns:
            Dictionary with 'pred_state_traj' (Tensor) and 'pred_state_one_step' (Tensor).
        """
        batch_size = state.shape[0]
        # Infer pred_horizon.
        # `action` is (batch * pred_horizon, action_dim_geom)
        # `next_state` (if provided by DAE) is also (batch * pred_horizon, state_dim_geom)
        if next_state is not None:
            if not isinstance(next_state, GeometricTensor):
                raise TypeError(f"next_state must be a GeometricTensor or None, got {type(next_state)}")
            pred_horizon = next_state.tensor.shape[0] // batch_size
        else:
            pred_horizon = action.tensor.shape[0] // batch_size

        if self.is_trainable:
            # self.forcast expects state: GeometricTensor, action: GeometricTensor
            # It returns a Tensor: (batch, n_steps + 1, state_dim_tensor)
            pred_state_traj_tensor = self.forcast(state=state, action=action, n_steps=pred_horizon)

            action_reshaped_tensor = action.tensor.view(batch_size, pred_horizon, self.action_dim)
            action_0_tensor = action_reshaped_tensor[:, 0, :]
            action_0_gt = self.action_type(action_0_tensor)
            one_step_evolved_state_gt = self.transfer_op(state) + self.control_op(action_0_gt)
            one_step_evolved_traj_tensor = torch.stack([state.tensor, one_step_evolved_state_gt.tensor], dim=1)

            out = dict(
                pred_state_traj=self.post_process_state(pred_state_traj_tensor),
                pred_state_one_step=self.post_process_state(one_step_evolved_traj_tensor),
            )
        else:
            # Non-trainable case
            pred_state_traj_tensor = self.forcast(state=state, action=action, n_steps=pred_horizon)
            out = dict(pred_state_traj=self.post_process_state(pred_state_traj_tensor))
            log.warning("Non-trainable ControlledEquivLinearDynamics.forward might not produce pred_state_one_step.")
        return out

    def forcast(self, state: GeometricTensor, action: GeometricTensor, n_steps: int = 1, **kwargs) -> Tensor:
        """
        Args:
            state: GT (batch, state_dim_geom) Initial state.
            action: GT (batch * n_steps, action_dim_geom) Flat sequence of actions.
        Returns:
            pred_state_traj: Tensor (batch, n_steps + 1, state_dim_tensor)
        """
        batch_size = state.shape[0]
        action_tensor_seq = action.tensor.view(batch_size, n_steps, self.action_dim)

        pred_state_traj_list_tensors = [state.tensor]
        current_state_gt = state

        for step in range(n_steps):
            current_action_tensor = action_tensor_seq[:, step, :]
            current_action_gt = self.action_type(current_action_tensor)

            if self.is_trainable:
                next_state_gt = self.transfer_op(current_state_gt) + self.control_op(current_action_gt)
            else:
                raise NotImplementedError("Non-trainable forecast for ControlledEquivLinearDynamics not implemented.")
            pred_state_traj_list_tensors.append(next_state_gt.tensor)
            current_state_gt = next_state_gt

        pred_state_traj_tensor = torch.stack(pred_state_traj_list_tensors, dim=1)
        return pred_state_traj_tensor

    def get_hparams(self):
        hparams = super().get_hparams()
        hparams.update(action_dim=self.action_dim, action_type_name=self.action_type.name if self.action_type else None)
        return hparams

    def pre_process_action(self, action: Tensor) -> Tensor:
        """Apply transformations to the action tensor before computing observable states.

        Args:
            action: (batch, action_dim) Action applied to the system.

        Returns:
            transformed_action: (batch, action_dim) tensor

        """
        # Assume that there is no change of basis
        transformed_action = action

        return transformed_action