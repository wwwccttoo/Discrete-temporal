import math
from typing import Optional

import gpytorch
import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.utils import subset_transform
from gpytorch import Module as GPyTorchModule
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from torch import Tensor
from torch.distributions import MultivariateNormal


class TemporalKernelB2P(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, epsilon=0.08, **kwargs):  # 0.01
        super().__init__(**kwargs)

        self.epsilon = epsilon

    # this is the kernel function
    def forward(self, x1, x2, **params):
        base = 1 - self.epsilon
        # calculate the distance between inputs
        exp = torch.abs(self.covar_dist(x1, x2, square_dist=False)) / 2
        out = torch.pow(base, exp)
        return out


class WienerKernel(gpytorch.kernels.Kernel):  # vorlesung von Phillip Henning
    is_stationary = False

    def __init__(self, c0, sigma_hat_squared=0.5, out_max=2, **kwargs):
        super().__init__(**kwargs)

        self.max_var = out_max
        self.sigma_hat_squared = sigma_hat_squared
        self.c0 = c0

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # d will always be 1, as it is the time dimenaion! Therefore we can squeeze the inputs
        if x1.ndim == 2:  # 'normal' mode
            x1, x2 = x1.squeeze(x1.ndim - 1), x2.squeeze(x2.ndim - 1)
            meshed_x1, meshed_x2 = torch.meshgrid(x1, x2)
            return self.evaluate_kernel(meshed_x1, meshed_x2)

        else:  # 'batch' mode
            # old
            # x1squeezed, x2squeezed = x1.squeeze(x1.ndim - 1), x2.squeeze(x2.ndim - 1)
            # t0 = time.time()
            # out = torch.empty((1, x1squeezed.shape[1], x2squeezed.shape[1]))
            # for batch in range(x1squeezed.shape[0]):
            #     x1_batch = x1squeezed[batch, :]
            #     x2_batch = x2squeezed[batch, :]
            #
            #     meshed_x1, meshed_x2 = torch.meshgrid(x1_batch, x2_batch)
            #     new_out = self.evaluate_kernel(meshed_x1, meshed_x2).unsqueeze(0)
            #
            #     out = torch.cat((out, new_out), dim=0)
            # out1 = out[1:, :, :]
            # print('Loop:', time.time() - t0)

            # t0 = time.time()
            meshed_x1 = torch.tile(x1, (1, 1, x2.shape[1]))
            meshed_x2 = torch.tile(x2.transpose(dim0=-2, dim1=-1), (1, x1.shape[1], 1))
            out = self.evaluate_kernel(meshed_x1, meshed_x2)
            return out

    def evaluate_kernel(self, meshed_x1, meshed_x2):
        step = torch.min(meshed_x1, meshed_x2) - self.c0
        out = step * self.sigma_hat_squared
        return out


class WienerKernel_learning(gpytorch.kernels.Kernel):  # vorlesung von Phillip Henning
    is_stationary = False

    def __init__(
        self,
        outputscale_prior: Optional[Prior] = None,
        outputscale_constraint: Optional[Interval] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if outputscale_constraint is None:
            outputscale_constraint = Positive()

        outputscale = torch.tensor(0.0)
        self.register_parameter(
            name="raw_outputscale", parameter=torch.nn.Parameter(outputscale)
        )

        if outputscale_prior is not None:
            if not isinstance(outputscale_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(outputscale_prior).__name__
                )
            self.register_prior(
                "outputscale_prior",
                outputscale_prior,
                self._outputscale_param,
                self._outputscale_closure,
            )

        self.register_constraint("raw_outputscale", outputscale_constraint)

    def _outputscale_param(self, m):
        return m.outputscale

    def _outputscale_closure(self, m, v):
        return m._set_outputscale(v)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(
            raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value)
        )

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # d will always be 1, as it is the time dimenaion! Therefore we can squeeze the inputs
        if x1.ndim == 2:  # 'normal' mode
            x1, x2 = x1.squeeze(x1.ndim - 1), x2.squeeze(x2.ndim - 1)
            meshed_x1, meshed_x2 = torch.meshgrid(x1, x2)
            return self.evaluate_kernel(meshed_x1, meshed_x2)

        else:  # 'batch' mode
            # old
            # x1squeezed, x2squeezed = x1.squeeze(x1.ndim - 1), x2.squeeze(x2.ndim - 1)
            # t0 = time.time()
            # out = torch.empty((1, x1squeezed.shape[1], x2squeezed.shape[1]))
            # for batch in range(x1squeezed.shape[0]):
            #     x1_batch = x1squeezed[batch, :]
            #     x2_batch = x2squeezed[batch, :]
            #
            #     meshed_x1, meshed_x2 = torch.meshgrid(x1_batch, x2_batch)
            #     new_out = self.evaluate_kernel(meshed_x1, meshed_x2).unsqueeze(0)
            #
            #     out = torch.cat((out, new_out), dim=0)
            # out1 = out[1:, :, :]
            # print('Loop:', time.time() - t0)

            # t0 = time.time()
            meshed_x1 = torch.tile(x1, (1, 1, x2.shape[1]))
            meshed_x2 = torch.tile(x2.transpose(dim0=-2, dim1=-1), (1, x1.shape[1], 1))
            out = self.evaluate_kernel(meshed_x1, meshed_x2)
            return out

    def evaluate_kernel(self, meshed_x1, meshed_x2):
        step = torch.min(meshed_x1, meshed_x2) + 1.0 / self.outputscale
        out = step * self.outputscale
        return out


class ConstantKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, constant=1, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return torch.ones_like(self.covar_dist(x1, x2))


# defined only for t>=0
class GeometricWienerKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, t, sigma=0.5, **kwargs):
        super().__init__(**kwargs)

        self.sigma = sigma
        self.c0 = t

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # d will always be 1, as it is the time dimenaion! Therefore we can squeeze it
        x1, x2 = x1.squeeze(x1.ndim - 1), x2.squeeze(x2.ndim - 1)

        if x1.ndim == 1:  # 'normal' mode
            meshed_x1, meshed_x2 = torch.meshgrid(x1, x2)
            return self.evaluate_kernel(meshed_x1, meshed_x2)
        else:  # batch mode

            out = torch.empty((1, x1.shape[1], x2.shape[1]))
            for batch in range(x1.shape[0]):
                x1_batch = x1[batch, :]
                x2_batch = x2[batch, :]

                meshed_x1, meshed_x2 = torch.meshgrid(x1_batch, x2_batch)
                new_out = self.evaluate_kernel(meshed_x1, meshed_x2).unsqueeze(0)

                out = torch.cat((out, new_out), dim=0)
            return out[1:, :, :]

    def evaluate_kernel(self, meshed_x1, meshed_x2):
        step = torch.min(meshed_x1, meshed_x2) - self.c0
        out = step * self.sigma**2
        return out


class TemporalKernelUI(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, epsilon_prior=0.08, **kwargs):  # 0.01
        super().__init__(**kwargs)

        self.epsilon = epsilon_prior

    # this is the kernel function
    def forward(self, x1, x2, **params):
        base = 1 - self.epsilon
        # calculate the distance between inputs
        exponent = torch.abs(self.covar_dist(x1, x2, square_dist=False)) / -2.0
        out = torch.exp(exponent)
        return out


class PosEncode(InputTransform, GPyTorchModule):
    r"""A transform that uses learned input warping functions."""

    def __init__(
        self,
        positional_emb_dim: int,
        reduce_to_one: bool = False,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        batch_shape: Optional[torch.Size] = None,
        tkwargs={"dtype": torch.float32, "device": "cpu"},
    ) -> None:
        r"""Initialize transform.

        Args:
            positional_emb_dim: Int, the positional embedding dimension, must be an even number.
            reduce_to_one: Bool, if we project the positional embedding to one variable (using NN like layer).
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            batch_shape: The batch shape.
        """
        super().__init__()
        self.positional_emb_dim = positional_emb_dim
        self.reduce_to_one = reduce_to_one
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.batch_shape = batch_shape or torch.Size([])
        self.tkwargs = tkwargs
        if len(self.batch_shape) > 0:
            # Note: this follows the gpytorch shape convention for lengthscales
            # There is ongoing discussion about the extra `1`.
            # TODO: update to follow new gpytorch convention resulting from
            # https://github.com/cornellius-gp/gpytorch/issues/1317
            batch_shape = self.batch_shape + torch.Size([1])
        else:
            batch_shape = self.batch_shape
        if self.reduce_to_one:
            # register weights
            self.register_parameter(
                "linear_W",
                torch.nn.Parameter(torch.zeros((positional_emb_dim, 1)).to(**tkwargs)),
            )
            torch.nn.init.kaiming_normal_(self.linear_W)
            # register bias
            self.register_parameter(
                "linear_B", torch.nn.Parameter(torch.zeros((1, 1)).to(**self.tkwargs))
            )
            torch.nn.init.kaiming_normal_(self.linear_B)
        self.register_parameter(
            "positive_increase_scaler",
            torch.nn.Parameter(5 * torch.ones((1, 1)).to(**self.tkwargs)),
        )
        self.register_constraint("positive_increase_scaler", Interval(0, 10))

    def get_positional_encoding(self, positional_emb_dim, idx_list):
        position = idx_list.reshape(-1).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, positional_emb_dim, 2)
            * (-math.log(10000.0) / positional_emb_dim)
        ).to(**self.tkwargs)
        # Introduce a learnable scaling factor that increases with positional index
        scaling_factor = torch.log(1 + position * self.positive_increase_scaler)
        pe = torch.zeros(len(position), 1, positional_emb_dim).to(**self.tkwargs)
        pe[:, 0, 0::2] = torch.sin(position * div_term) * scaling_factor
        pe[:, 0, 1::2] = torch.cos(position * div_term) * scaling_factor
        return pe.squeeze(1)  # (len, dim)

    @subset_transform
    def transform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs.

        Args:
            X: A `input_batch_shape x (batch_shape) x n x d`-dim tensor of inputs.
                batch_shape here can either be self.batch_shape or 1's such that
                it is broadcastable with self.batch_shape if self.batch_shape is set.

        Returns:
            A `input_batch_shape x (batch_shape) x n x d`-dim tensor of transformed
                inputs.
        """
        positions = X.reshape(-1, X.shape[-1])[..., -1]
        PEs = self.get_positional_encoding(self.positional_emb_dim, positions)
        if self.reduce_to_one:
            # maybe need to pull the distribution back?
            # apply abs and a scaler factor, (almost) making sure the scalar is increasing
            # as the index increases
            PEs = PEs.abs() @ torch.abs(self.linear_W) / math.sqrt(
                PEs.shape[-1]
            ) + torch.abs(self.linear_B)
            PEs = torch.nn.functional.gelu(PEs).reshape(X.shape[:-1] + (1,))
        else:
            PEs = PEs.reshape(X.shape[:-1] + (self.positional_emb_dim,))

        return torch.cat((X[..., :-1], PEs), dim=-1)

    @property
    def _k(self) -> MultivariateNormal:
        """Returns a MultivariateNormal distribution."""
        return MultivariateNormal(
            loc=torch.zeros((self.positional_emb_dim + 1)),
            covariance_matrix=torch.eye((self.positional_emb_dim + 1)),
        )
