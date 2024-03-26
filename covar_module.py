from typing import Optional, Tuple

import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor

from custom_kernels import (
    ConstantKernel,
    TemporalKernelB2P,
    WienerKernel,
    WienerKernel_learning,
)


def get_matern_kernel_with_gamma_prior(
    ard_num_dims: int,
    batch_shape: Optional[torch.Size] = None,
    active_dims=None,
) -> ScaleKernel:
    r"""Constructs the Scale-Matern kernel that is used by default by
    several models. This uses a Gamma(3.0, 6.0) prior for the lengthscale
    and a Gamma(2.0, 0.15) prior for the output scale.
    """
    return ScaleKernel(
        base_kernel=MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ),
        batch_shape=batch_shape,
        outputscale_prior=GammaPrior(2.0, 0.15),
    )


def get_batch_dimensions(
    train_X: Tensor, train_Y: Tensor
) -> Tuple[torch.Size, torch.Size]:
    r"""Get the raw batch shape and output-augmented batch shape of the inputs.

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
            features.
        train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
            training observations.

    Returns:
        2-element tuple containing

        - The `input_batch_shape`
        - The output-augmented batch shape: `input_batch_shape x (m)`
    """
    input_batch_shape = train_X.shape[:-2]
    aug_batch_shape = input_batch_shape
    num_outputs = train_Y.shape[-1]
    if num_outputs > 1:
        aug_batch_shape += torch.Size([num_outputs])
    return input_batch_shape, aug_batch_shape


def get_spatio_temp_kernel(
    train_X,
    train_Y,
    type_of_forgetting,
    forgetting_factor=0.2,
    outputscale_hyperprior_temporal=None,
    outputscale_constraint_temporal=None,
):
    # the default temporal dimension is the last dimension
    input_batch_shape, aug_batch_shape = get_batch_dimensions(train_X, train_Y)
    spatio_dims = train_X.shape[-1] - 1
    spatio_kernel = get_matern_kernel_with_gamma_prior(
        ard_num_dims=spatio_dims,
        batch_shape=aug_batch_shape,
        active_dims=range(spatio_dims),
    )
    # specify forgetting strategy
    if type_of_forgetting == "UI":  # wiener prozess kernel
        forgetting_factor = forgetting_factor
        sigma_w_squared = forgetting_factor / spatio_kernel.outputscale.clone().detach()
        c0 = -1 / sigma_w_squared
        temporal_kernel = WienerKernel(
            c0=c0,  # start at t= -10 to have higher flexibility in the mean
            sigma_hat_squared=sigma_w_squared,
            active_dims=spatio_dims,
        )
        spatio_kernel.raw_outputscale.requires_grad = False

    elif type_of_forgetting == "UI_learning":
        temporal_kernel = WienerKernel_learning(
            active_dims=spatio_dims
        )  # maybe consider a prior here?

    elif type_of_forgetting == "B2P_SE":  # squared-exponential kernel
        temporal_kernel = ScaleKernel(
            RBFKernel(active_dims=spatio_dims),
            outpurtscale_prior=outputscale_hyperprior_temporal,
            outputscale_constraint=outputscale_constraint_temporal,
        )
        temporal_kernel.base_kernel.lengthscale = 52.0
        temporal_kernel.outputscale = 5

    elif type_of_forgetting == "B2P_OU":  # ohrnstrein-uhlenbeck kernel
        temporal_kernel = TemporalKernelB2P(
            epsilon=forgetting_factor, active_dims=spatio_dims
        )
        spatio_kernel.outputscale = 1
        spatio_kernel.raw_outputscale.requires_grad = False
    elif type_of_forgetting == "vanilla_with_index":
        spatio_dims = train_X.shape[-1]
        spatio_kernel = get_matern_kernel_with_gamma_prior(
            ard_num_dims=spatio_dims,
            batch_shape=aug_batch_shape,
            active_dims=range(spatio_dims),
        )
        return spatio_kernel
    elif type_of_forgetting is None:
        temporal_kernel = ConstantKernel(active_dims=spatio_dims)
    else:
        raise NotImplementedError(
            "Only Uncertainty-Injection (“UI“) and "
            'Back-2-Prior forgetting ("B2P_SE", "B2P_OU") are implemented.'
        )
    return spatio_kernel * temporal_kernel
