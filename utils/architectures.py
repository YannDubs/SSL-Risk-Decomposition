from functools import partial

import torch
from torch import nn
from utils.helpers import weights_init


def get_Architecture(architecture: str, **kwargs):
    """Return the (uninstantiated) correct architecture.

    Parameters
    ----------
    architecture : {"mlp", "linear"}.
        Architecture to return

    kwargs :
        Additional arguments to the Module.

    Return
    ------
    Architecture : uninstantiated nn.Module
        Architecture that can be instantiated by `Architecture(in_shape, out_shape)`
    """
    if architecture == "mlp":
        return partial(MLP, **kwargs)

    elif architecture == "linear":
        return partial(Linear, **kwargs)

    else:
        raise ValueError(f"Unknown architecture={architecture}.")


class Linear(nn.Linear):
    """Linear layer

    Parameters
    ----------
    in_dim : int

    out_dim : int

    is_normalize : bool, optional
        Whether to use a batchnorm layer before the input.

    is_l2_normalize : bool, optional
        Whether to use a batchnorm layer before the input.

    kwargs :
        Additional arguments to `torch.nn.Linear`.
    """

    def __init__(
        self, in_dim: int, out_dim: int, is_normalize : bool = True, is_l2_normalize: bool=False, **kwargs
    ) -> None:
        super().__init__(in_features=in_dim, out_features=out_dim, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_l2_normalize = is_l2_normalize
        self.is_normalize = is_normalize
        self.normalizer = nn.BatchNorm1d(self.in_dim) if is_normalize else nn.Identity()

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_l2_normalize:
            X = nn.functional.normalize(X, dim=-1)

        return super().forward(self.normalizer(X))

    def reset_parameters(self):
        weights_init(self)


class MLP(nn.Module):
    """Multi Layer Perceptron.

    Parameters
    ----------
    in_dim : int

    out_dim : int

    hid_dim : int, optional
        Number of hidden neurones.

    n_hid_layers : int, optional
        Number of hidden layers.

    is_normalize : bool, optional
        Whether to use batchnorm layers.

    dropout_p : float, optional
        Dropout rate.

    is_skip_hidden : bool, optional
        Whether to skip all the hidden layers with a residual connection.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_hid_layers: int = 2,
        hid_dim: int = 2048,
        is_normalize: bool = True,
        dropout_p: float = 0,
        is_skip_hidden: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid_layers = n_hid_layers
        self.hid_dim = hid_dim
        Activation = nn.ReLU
        Dropout = nn.Dropout if dropout_p > 0 else nn.Identity
        Norm = nn.BatchNorm1d if is_normalize else nn.Identity
        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        bias_hidden = is_normalize
        self.is_skip_hidden = is_skip_hidden

        self.pre_block = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias_hidden),
            Norm(hid_dim),
            Activation(),
            Dropout(p=dropout_p),
        )
        layers = []
        # start at 1 because pre_block
        for _ in range(1, n_hid_layers):
            layers += [
                nn.Linear(hid_dim, hid_dim, bias=bias_hidden),
                Norm(hid_dim),
                Activation(),
                Dropout(p=dropout_p),
            ]
        self.hidden_block = nn.Sequential(*layers)
        self.post_block = nn.Linear(hid_dim, out_dim)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.pre_block(X)

        if self.is_skip_hidden:
            # use a residual connection for all the hidden block
            X = self.hidden_block(X) + X
        else:
            X = self.hidden_block(X)

        X = self.post_block(X)
        return X

    def reset_parameters(self):
        weights_init(self)
