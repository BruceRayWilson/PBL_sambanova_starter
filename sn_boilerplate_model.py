"""boilerplate model definition."""
#import argparse
#import sys
#from typing import Tuple


import torch
import torch.nn as nn
#import torchvision

from sambaflow import samba

#import sambaflow.samba.utils as utils
#from sambaflow.samba.utils.argparser import parse_app_args
#from sambaflow.samba.utils.pef_utils import get_pefmeta
#from sambaflow.samba.utils.dataset.mnist import dataset_transform


class FFN(nn.Module):
    """Feed Forward Network."""

    def __init__(self, num_features: int, ffn_dim_1: int, ffn_dim_2: int) -> None:
        """Initialize the class."""
        super().__init__()
        self.gemm1 = nn.Linear(num_features, ffn_dim_1, bias=False)
        self.relu = nn.ReLU()
        self.gemm2 = nn.Linear(ffn_dim_1, ffn_dim_2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Step forward."""
        out = self.gemm1(x)
        out = self.relu(out)
        out = self.gemm2(out)
        return out


class LogReg(nn.Module):
    """Logreg class."""

    def __init__(self, num_features: int, num_classes: int):
        """Initialize the class."""
        super().__init__()
        self.lin_layer = nn.Linear(in_features=num_features, out_features=num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Step forward."""
        out = self.lin_layer(inputs)
        loss = self.criterion(out, targets)
        return loss, out


class FFNLogReg(nn.Module):
    """Feed Forward Network + LogReg."""

    def __init__(self, num_features: int, ffn_embedding_size: int, embedding_size: int, num_classes: int) -> None:
        """Initialize the class."""
        super().__init__()
        self.ffn = FFN(num_features, ffn_embedding_size, embedding_size)
        self.logreg = LogReg(embedding_size, num_classes)
        self._init_params()

    def _init_params(self) -> None:
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Step forward."""
        out = self.ffn(inputs)
        loss, out = self.logreg(out, targets)
        return loss, out

    @staticmethod
    def get_fake_inputs(args):
        """
        Get fake inputs.

        The size of the inputs are required for the SambaNova compiler.

        Args:
            args: CLI arguments.

        Outputs:
            X_randn: A Samba tensor representing the correct shape for model inputs.
            Y_randint: A Samba tensor representing the correct shape for model outputs.
        """
        X_randn = samba.randn(args.batch_size, args.num_features, name='image', batch_dim=0).bfloat16().float()

        low_inclusive = 0
        high_exclusive = args.num_classes

        # The size/shape of the output tensor.
        size = (args.batch_size, )
        Y_randint = samba.randint(  low_inclusive,
                                    high_exclusive,
                                    size,
                                    name='label',
                                    batch_dim=0)

        return X_randn, Y_randint
