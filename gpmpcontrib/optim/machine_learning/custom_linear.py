import torch
import torch.nn as nn
from torch.nn import functional as _, init
import math

class CustomLinear(nn.Linear):

    def __init__(self, *args, generator=torch.Generator(), **kwargs):
        self.generator = generator

        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5), generator=self.generator)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound, generator=self.generator)
