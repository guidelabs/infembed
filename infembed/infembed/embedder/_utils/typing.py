#!/usr/bin/env python3

from typing import List, Tuple, TYPE_CHECKING, TypeVar, Union

from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 8):
        from typing import Literal  # noqa: F401
    else:
        from typing_extensions import Literal  # noqa: F401
else:
    Literal = {True: bool, False: bool, (True, False): bool}