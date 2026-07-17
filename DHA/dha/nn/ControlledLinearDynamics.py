import logging
from typing import Optional, Protocol, Union, Dict

import torch
from escnn.group import Representation
from torch import Tensor

from dha.nn.LinearDynamics import LinearDynamics
from dha.utils.linear_algebra import full_rank_lstsq

log = logging.getLogger(__name__)


class DmdSolver(Protocol):
    def __call__(self, X: Tensor, X_prime: Tensor, **kwargs) -> Tensor:
        """Compute the least squares solution of the linear system X' = X·A.

        Args:
            X: (|x|, n_samples) Data matrix of the initial states.
            Y: (|y|, n_samples) Data matrix of the next states.

        Returns:
            A: (|y|, |x|) Least squares solution of the linear system `X' = A·X`.

        """
        ...


class ControlledLinearDynamics(LinearDynamics):
    def __init__(
        self,
        state_dim: Optional[int] = None,
        state_rep: Optional[Representation] = None,
        action_dim: Optional[int] = None,
        action_rep: Optional[Representation] = None,
        bias: bool = True,
        dmd_algorithm: Optional[DmdSolver] = None,
        dt: Optional[Union[float, int]] = 1,
        trainable=False,
        init_mode: str = "stable",
        **markov_dyn_kwargs,
    ):

        assert action_dim is not None or action_rep is not None, "Either action_dim or action_rep must be provided"
        self.action_rep: Representation = action_rep
        self.action_dim = action_dim if action_dim is not None else action_rep.size

        super().__init__(state_dim=state_dim, state_rep=state_rep, bias=bias, dmd_algorithm=dmd_algorithm, dt=dt, trainable=trainable, init_mode=init_mode, **markov_dyn_kwargs)

        # Variables for non-training mode
        if not trainable:
            self.control_op = None
            self.control_op_bias = None
        else:
            self.control_op = self.build_control_linear_map()
            # Initialize weights of the linear layer such that it represents a stable system
            self.reset_control_parameters(init_mode=init_mode)

    def forward(self, state: Tensor, action: Tensor, next_state: Optional[Tensor] = None, **kwargs) -> [Dict[str, Tensor]]:
        pred_horizon = next_state.shape[1] if next_state is not None else 1
        pre_processed_state = self.pre_process_state(state=state)
        pre_processed_action = self.pre_process_action(action=action)

        if self.is_trainable:
            state_traj = self.pre_process_state(state=state, next_state=next_state)
            one_step_evolved_traj = self.transfer_op(state_traj)
            pred_state_traj = self.forcast(state=pre_processed_state, action=pre_processed_action, n_steps=pred_horizon)
            out = dict(
                pred_state_traj=self.post_process_state(pred_state_traj),
                pred_state_one_step=self.post_process_state(one_step_evolved_traj),
            )
        else:
            pred_state_traj = self.forcast(state=pre_processed_state, n_steps=pred_horizon)
            out = dict(pred_state_traj=self.post_process_state(pred_state_traj))
        return out

    def forcast(self, state: Tensor, action: Tensor, n_steps: int = 1, **kwargs) -> Tensor:
        """Predict the next `n_steps` states of the system.

        Args:
            state: (batch, state_dim) Initial state of the system.
            action: (batch, n_steps, action_dim) Action and future actions applied to the system.
            n_steps: (int) Number of steps to predict.

        Returns:
            pred_state_traj: (batch, n_steps + 1, state_dim)

        """
        batch, state_dim = state.shape
        action_dim = action.shape[2]
        assert state_dim == self.state_dim and action_dim == self.action_dim

        # Use the transfer operator and control operator to compute the maximum likelihood prediction of the future trajectory
        pred_state_traj = [state]

        for step in range(n_steps):
            # Compute the next state prediction s_t+1 = A @ s_t + B u_t
            current_state = pred_state_traj[-1]
            current_action = action[:, step, :]
            if self.is_trainable:
                next_obs_state = self.transfer_op(current_state) + self.control_op(current_action)
            else:
                transfer_op, bias = self.get_transfer_op()
                control_op, control_bias = self.get_control_op()
                if bias is not None:
                    next_obs_state = (transfer_op @ current_state.T + bias + control_op @ current_action.T + control_bias).T
                else:
                    next_obs_state = (transfer_op @ current_state.T + control_op @ current_action.T).T
            pred_state_traj.append(next_obs_state)

        pred_state_traj = torch.stack(pred_state_traj, dim=1)
        assert pred_state_traj.shape == (batch, n_steps + 1, state_dim)
        return pred_state_traj

    def get_transfer_op(self):
        if self.is_trainable:
            raise RuntimeError("This model was initialized as trainable")
        else:
            transfer_op = self.transfer_op
            bias = self.transfer_op_bias
            if transfer_op is None:
                raise RuntimeError("The transfer operator not approximated yet. Call `approximate_transfer_operator`")
        return transfer_op, bias

    def get_control_op(self):
        if self.is_trainable:
            raise RuntimeError("This model was initialized as trainable")
        else:
            control_op = self.control_op
            bias = self.control_op_bias
            if control_op is None:
                raise RuntimeError("The control operator not approximated yet. Call `approximate_control_operator`")
        return control_op, bias

    def update_transfer_op(self, X: Tensor, X_prime: Tensor) -> Dict[str, Tensor]:
        """Use a DMD algorithm to update the empirical transfer operator
        Args:
            X: (state_dim, n_samples) Data matrix of states at time `t`.
            X_prime: (state_dim, n_samples) Data matrix of the states at time `t + dt`.

        Returns:
            metrics (dict): Dictionary of metrics computed during the update.

        """
        if self.is_trainable:
            raise RuntimeError("This model was initialized as trainable")

        assert X.shape == X_prime.shape, f"X: {X.shape}, X_prime: {X_prime.shape}"
        assert X.shape[0] == self.state_dim, f"Invalid state dimension {X.shape[0]} != {self.state_dim}"

        A, B = self.dmd_algorithm(X=X, Y=X_prime, bias=self.bias)
        if self.bias:
            rec_error = torch.nn.functional.mse_loss(A @ X + B, X_prime)
        else:
            rec_error = torch.nn.functional.mse_loss(A @ X, X_prime)

        self.transfer_op = A
        self.transfer_op_bias = B
        return dict(
            solution_op_rank=torch.linalg.matrix_rank(A.detach()).to(torch.float),
            solution_op_cond_num=torch.linalg.cond(A.detach()).to(torch.float),
            solution_op_error=rec_error.detach().to(torch.float),
        )

    def build_control_linear_map(self) -> torch.nn.Linear:
        return torch.nn.Linear(self.action_dim, self.state_dim, bias=self.bias)

    def get_hparams(self):
        main_params = dict(state_dim=self.state_dim, action_dim=self.action_dim, trainable=self.is_trainable)
        return main_params

    def reset_control_parameters(self, init_mode: str):
        if init_mode == "stable":
            self.control_op.weight.data = torch.zeros(self.state_dim, self.action_dim)
            if self.bias:
                self.control_op.bias.data = torch.zeros(self.state_dim)
        else:
            raise NotImplementedError(f"Eival init mode {init_mode} not implemented")
        log.info(f"Eigenvalues initialization to {init_mode}")

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
