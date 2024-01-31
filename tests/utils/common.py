import functools
import inspect
from typing import Callable, List, Optional
import torch
from parameterized.parameterized import param
from parameterized import parameterized


def assertTensorAlmostEqual(test, actual, expected, delta=0.0001, mode="sum"):
    assert isinstance(actual, torch.Tensor), (
        "Actual parameter given for " "comparison must be a tensor."
    )
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    assert (
        actual.shape == expected.shape
    ), f"Expected tensor with shape: {expected.shape}. Actual shape {actual.shape}."
    actual = actual.cpu()
    expected = expected.cpu()
    if mode == "sum":
        test.assertAlmostEqual(
            torch.sum(torch.abs(actual - expected)).item(), 0.0, delta=delta
        )
    elif mode == "max":
        # if both tensors are empty, they are equal but there is no max
        if actual.numel() == expected.numel() == 0:
            return

        if actual.size() == torch.Size([]):
            test.assertAlmostEqual(
                torch.max(torch.abs(actual - expected)).item(), 0.0, delta=delta
            )
        else:
            for index, (input, ref) in enumerate(zip(actual, expected)):
                almost_equal = abs(input - ref) <= delta
                if hasattr(almost_equal, "__iter__"):
                    almost_equal = almost_equal.all()
                #print(index, input / ref)
                assert (
                    almost_equal
                ), "Values at index {}, {} and {}, differ more than by {}".format(
                    index, input, ref, delta
                )
    else:
        raise ValueError("Mode for assertion comparison must be one of `max` or `sum`.")
    

def generate_test_name(
    testcase_func: Callable,
    param_num: str,
    param: param,
    args_to_skip: Optional[List[str]] = None,
) -> str:
    """
    Creates human readable names for parameterized tests
    """

    if args_to_skip is None:
        args_to_skip = []
    param_strs = []

    func_param_names = list(inspect.signature(testcase_func).parameters)
    # skip the first 'self' parameter
    if func_param_names[0] == "self":
        func_param_names = func_param_names[1:]

    for i, arg in enumerate(param.args):
        if func_param_names[i] in args_to_skip:
            continue
        if isinstance(arg, bool):
            if arg:
                param_strs.append(func_param_names[i])
        else:
            args_str = str(arg)
            if args_str.isnumeric():
                param_strs.append(func_param_names[i])
            param_strs.append(args_str)
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(param_strs)),
    )


def build_test_name_func(args_to_skip: Optional[List[str]] = None):
    """
    Returns function to generate human readable names for parameterized tests
    """

    return functools.partial(generate_test_name, args_to_skip=args_to_skip)