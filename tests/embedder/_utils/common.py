import functools
import inspect
from typing import Callable, List, Optional, Union
from infembed.embedder._utils.common import _format_inputs_dataset
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from parameterized.parameterized import param
from parameterized import parameterized


class EmbedderConstructor:
    f"""
    A constructor for an `Embedder` whose initialization arguments are the model and
    related arguments.  In practice it just means arguments not related to them
    are stored, i.e. acts like a `functools.partial`.
    """
    def __init__(self, constructor, name: Optional[str] = None, **kwargs):
        self._constructor = functools.partial(constructor, **kwargs)
        self.kwargs = kwargs
        self.name = name if name is not None else self._default_name()

    def _default_name(self):
        return f"{self._constructor.func.__name__}_{self.kwargs}"
    
    def __repr__(self):
        return self.name
    
    def __call__(self, model: Module, **kwargs):
        return self._constructor(model=model, **kwargs)
    

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
    

def _test_compare_implementations(
    test,
    model_type: str,
    embedder_constructor_1: Callable,
    embedder_constructor_2: Callable,
    delta: float,
    unpack_inputs: bool,
    use_gpu: Union[bool, str],
) -> None:
    """
    checks that 2 implementations of `EmbedderBase` return the same output, where the
    output is the same if the dot-product of embeddings between different test
    examples iss the same. this is a helper used by other tests. the implementations
    are compared using the same data, but the model and saved checkpoints can be
    different, and is specified using the `model_type` argument.
    """
    (
        net,
        train_dataset,
        test_samples,
        test_labels,
    ) = get_random_model_and_data(
        unpack_inputs,
        use_gpu=use_gpu,
        model_type=model_type,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=5)

    criterion = nn.MSELoss(reduction="none")

    embedder_1 = embedder_constructor_1(
        model=net,
        loss_fn=criterion,
    )
    embedder_1.fit(train_dataloader)

    embedder_2 = embedder_constructor_2(
        model=net,
        loss_fn=criterion,
    )
    embedder_2.fit(train_dataloader)

    # compute pairwise dot-products for both implementations' embeddings
    test_dataloader = _format_inputs_dataset(
        (test_samples, test_labels) if not unpack_inputs else (*test_samples, test_labels)
    )
    embeddings_1 = embedder_1.predict(test_dataloader)
    embeddings_2 = embedder_2.predict(test_dataloader)

    influences_1 = embeddings_1 @ embeddings_1.T
    influences_2 = embeddings_2 @ embeddings_2.T

    assertTensorAlmostEqual(test, influences_1, influences_2, delta=delta, mode="sum")
        

def _move_sample_to_cuda(samples):
    return [s.cuda() for s in samples]


class ExplicitDataset(Dataset):
    def __init__(self, samples, labels, use_gpu=False) -> None:
        self.samples, self.labels = samples, labels
        if use_gpu:
            self.samples = (
                _move_sample_to_cuda(self.samples)
                if isinstance(self.samples, list)
                else self.samples.cuda()
            )
            self.labels = self.labels.cuda()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.labels[idx])


class UnpackDataset(Dataset):
    def __init__(self, samples, labels, use_gpu=False) -> None:
        self.samples, self.labels = samples, labels
        if use_gpu:
            self.samples = (
                _move_sample_to_cuda(self.samples)
                if isinstance(self.samples, list)
                else self.samples.cuda()
            )
            self.labels = self.labels.cuda()

    def __len__(self) -> int:
        return len(self.samples[0])

    def __getitem__(self, idx):
        """
        The signature of the returning item is: List[List], where the contents
        are: [sample_0, sample_1, ...] + [labels] (two lists concacenated).
        """
        return [lst[idx] for lst in self.samples] + [self.labels[idx]]
    

class BasicLinearNet(nn.Module):
    def __init__(self, in_features, hidden_nodes, out_features) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, out_features)

    def forward(self, input):
        x = torch.tanh(self.linear1(input))
        return torch.tanh(self.linear2(x))


class MultLinearNet(nn.Module):
    def __init__(self, in_features, hidden_nodes, out_features, num_inputs) -> None:
        super().__init__()
        self.pre = nn.Linear(in_features * num_inputs, in_features)
        self.linear1 = nn.Linear(in_features, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, out_features)

    def forward(self, *inputs):
        """
        The signature of inputs is List[torch.Tensor],
        where torch.Tensor has the dimensions [num_inputs x in_features].
        It first concacenates the list and a linear layer to reduce the
        dimension.
        """
        inputs = self.pre(torch.cat(inputs, dim=-1))
        x = torch.tanh(self.linear1(inputs))
        return torch.tanh(self.linear2(x))
    

class LinearWithConv2dNet(nn.Module):
    """
    network whose input and output is as in `BasicLinearNet`, but includes a `Conv2D`
    layer.
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, 48, bias=False)
        self.conv = nn.Conv2d(3, 1, 4, 2, 1)
        self.linear2 = nn.Linear(4, out_features, bias=False)

    def forward(self, input):
        x = torch.tanh(self.linear1(input))  # 48
        x = x.view(-1, 3, 4, 4)  # 3, 4, 4
        x = self.conv(x)  # 1, 2, 2
        x = x.view(-1, 4)  # 4
        return torch.tanh(self.linear2(x))


class MultLinearWithConv2dNet(nn.Module):
    """
    network whose input and output is as in `MultiLinearNet`, but includes a `Conv2D`
    layer.
    """

    def __init__(self, in_features, out_features, num_inputs) -> None:
        super().__init__()
        self.pre = nn.Linear(in_features * num_inputs, in_features)
        self.linear1 = nn.Linear(in_features, 48)
        self.conv = nn.Conv2d(3, 1, 4, 2, 1)
        self.linear2 = nn.Linear(4, out_features)

    def forward(self, *inputs):
        inputs = self.pre(torch.cat(inputs, dim=1))
        x = torch.tanh(self.linear1(inputs))  # 48
        x = x.view(-1, 3, 4, 4)  # 3, 4, 4
        x = self.conv(x)  # 1, 2, 2
        x = x.view(-1, 4)  # 4
        return torch.tanh(self.linear2(x))
    

class Linear(nn.Module):
    """
    a wrapper around `nn.Linear`, with purpose being to have an analogue to
    `UnpackLinear`, with both's only parameter being 'linear'. "infinitesimal"
    influence (i.e. that calculated by `InfluenceFunctionBase` implementations) for
    this simple module can be analytically calculated, so its purpose is for testing
    those implementations.
    """

    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input):
        return self.linear(input)


class UnpackLinear(nn.Module):
    """
    the analogue of `Linear` which unpacks inputs, serving the same purpose.
    """

    def __init__(self, in_features, out_features, num_inputs) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features * num_inputs, out_features, bias=False)

    def forward(self, *inputs):
        return self.linear(torch.cat(inputs, dim=1))
    

def _wrap_model_in_dataparallel(net):
    alt_device_ids = [0] + [x for x in range(torch.cuda.device_count() - 1, 0, -1)]
    net = net.cuda()
    return torch.nn.DataParallel(net, device_ids=alt_device_ids)
        

def get_random_model_and_data(
    unpack_inputs,
    use_gpu=False,
    model_type="random",
):
    """
    returns a model, training data, and optionally data for computing the hessian
    (needed for `InfluenceFunctionBase` implementations) as features / labels, and
    optionally test data as features / labels.

    the data is always generated the same way. however depending on `model_type`,
    a different model and checkpoints are returned.
    - `model_type='random'`: the model is a 2-layer NN, and several checkpoints are
    generated
    - `model_type='trained_linear'`: the model is a linear model, and assumed to be
    eventually trained to optimality. therefore, we find the optimal parameters, and
    save a single checkpoint containing them. the training is done using the Hessian
    data, because the purpose of training the model is so that the Hessian is positive
    definite. since the Hessian is calculated using the Hessian data, it should be
    used for training. since it is trained to optimality using the Hessian data, we can
    guarantee that the Hessian is positive definite, so that different
    implementations of `InfluenceFunctionBase` can be more easily compared. (if the
    Hessian is not positive definite, we drop eigenvectors corresponding to negative
    eigenvalues. since the eigenvectors dropped in `ArnoldiInfluence` differ from those
    in `NaiveInfluenceFunction` due to the formers' use of Arnoldi iteration, we should
    only use models / data whose Hessian is positive definite, so that no eigenvectors
    are dropped). in short, this model / data are suitable for comparing different
    `InfluenceFunctionBase` implementations.
    - `model_type='trained_NN'`: the model is a 2-layer NN, and trained (not
    necessarily) to optimality using the Hessian data. since it is trained, for same
    reasons as for `model_type='trained_linear`, different implementations of
    `InfluenceFunctionBase` can be more easily compared, due to lack of numerical
    issues.
    - `model_type='conv'`: linear model with `Conv2D` layer in the middle.
    - `model_type='seq'`: input is 3D
    - `model_type='one_layer_linear'`: 1-layer linear model

    `use_gpu` can either be
    - `False`: returned model is on cpu
    - `'cuda'`: returned model is on gpu
    - `'cuda_data_parallel``: returned model is a `DataParallel` model, and on cpu
    The need to differentiate between `'cuda'` and `'cuda_data_parallel'` is that sometimes
    we may want to test a model that is on cpu, but is *not* wrapped in `DataParallel`.
    """
    in_features, hidden_nodes = 5, 4
    num_inputs = 2

    # generate data. regardless the model, the data is always generated the same way
    # the only exception is if the `model_type` is 'trained_linear', i.e. a simple
    # linear regression model. this is a simple model, and for simplicity, the
    # number of `out_features` is 1 in this case.
    if model_type in ["trained_linear", "one_layer_linear"]:
        out_features = 1
    else:
        out_features = 3

    # if making sequence model, insert sequence dim
    if model_type == "seq":
        extra_shape = (6,)
    else:
        extra_shape = tuple()

    num_samples = 50
    num_train = 32
    all_labels = torch.normal(1, 2, (num_samples, *extra_shape, out_features)).double()
    all_labels = all_labels.cuda() if use_gpu else all_labels
    train_labels = all_labels[:num_train]
    test_labels = all_labels[num_train:]

    if unpack_inputs:
        all_samples = [
            torch.normal(0, 1, (num_samples, *extra_shape, in_features)).double()
            for _ in range(num_inputs)
        ]
        all_samples = (
            _move_sample_to_cuda(all_samples)
            if isinstance(all_samples, list) and use_gpu
            else all_samples.cuda()
            if use_gpu
            else all_samples
        )
        train_samples = [ts[:num_train] for ts in all_samples]
        test_samples = [ts[num_train:] for ts in all_samples]
    else:
        all_samples = torch.normal(
            0, 1, (num_samples, *extra_shape, in_features)
        ).double()
        all_samples = (
            _move_sample_to_cuda(all_samples)
            if isinstance(all_samples, list) and use_gpu
            else all_samples.cuda()
            if use_gpu
            else all_samples
        )
        train_samples = all_samples[:num_train]
        test_samples = all_samples[num_train:]

    dataset = (
        ExplicitDataset(train_samples, train_labels, use_gpu)
        if not unpack_inputs
        else UnpackDataset(train_samples, train_labels, use_gpu)
    )

    if model_type == "random":
        net = (
            BasicLinearNet(in_features, hidden_nodes, out_features)
            if not unpack_inputs
            else MultLinearNet(in_features, hidden_nodes, out_features, num_inputs)
        ).double()

        net.linear1.weight.data = torch.normal(
            3, 4, (hidden_nodes, in_features)
        ).double()
        net.linear2.weight.data = torch.normal(
            5, 6, (out_features, hidden_nodes)
        ).double()
        if unpack_inputs:
            net.pre.weight.data = torch.normal(
                3, 4, (in_features, in_features * num_inputs)
            ).double()
        net_adjusted = (
            _wrap_model_in_dataparallel(net)
            if use_gpu == "cuda_data_parallel"
            else (net.to(device="cuda") if use_gpu == "cuda" else net)
        )

    elif model_type in ["conv", "seq", "one_layer_linear"]:
        if model_type == "conv":
            net = (
                LinearWithConv2dNet(in_features, out_features)
                if not unpack_inputs
                else MultLinearWithConv2dNet(in_features, out_features, num_inputs)
            ).double()
        elif model_type == "seq":
            net = (
                BasicLinearNet(in_features, hidden_nodes, out_features)
                if not unpack_inputs
                else MultLinearNet(in_features, hidden_nodes, out_features, num_inputs)
            ).double()
        elif model_type == "one_layer_linear":
            net = (
                Linear(in_features, out_features)
                if not unpack_inputs
                else UnpackLinear(in_features, out_features, num_inputs)
            ).double()

        net_adjusted = (
            _wrap_model_in_dataparallel(net)
            if use_gpu == "cuda_data_parallel"
            else (net.to(device="cuda") if use_gpu == "cuda" else net)
        )

    elif model_type == "seq":
        net = (
            BasicLinearNet(in_features, hidden_nodes, out_features)
            if not unpack_inputs
            else MultLinearNet(in_features, hidden_nodes, out_features, num_inputs)
        ).double()

        net_adjusted = (
            _wrap_model_in_dataparallel(net)
            if use_gpu == "cuda_data_parallel"
            else (net.to(device="cuda") if use_gpu == "cuda" else net)
        )

    elif model_type == "trained_linear":
        net = (
            Linear(in_features)
            if not unpack_inputs
            else UnpackLinear(in_features, out_features, num_inputs)
        ).double()

        # regardless of `unpack_inputs`, the model is a linear regression, so that we can get
        # the optimal trained parameters via least squares

        # turn input into a single tensor for use by least squares
        tensor_train_samples = (
            train_samples if not unpack_inputs else torch.cat(train_samples, dim=1)
        )
        # run least squares to get optimal trained parameters
        theta = torch.linalg.lstsq(
            train_labels,
            tensor_train_samples,
        ).solution
        # the first `n` rows of `theta` contains the least squares solution, where `n` is the
        # number of features in `tensor_train_samples`
        theta = theta[: tensor_train_samples.shape[1]]

        net.linear.weight.data = theta.contiguous()
        net_adjusted = (
            _wrap_model_in_dataparallel(net)
            if use_gpu == "cuda_data_parallel"
            else (net.to(device="cuda") if use_gpu == "cuda" else net)
        )

    elif model_type == "trained_NN":
        net = (
            BasicLinearNet(in_features, hidden_nodes, out_features)
            if not unpack_inputs
            else MultLinearNet(in_features, hidden_nodes, out_features, num_inputs)
        ).double()

        net_adjusted = (
            _wrap_model_in_dataparallel(net)
            if use_gpu == "cuda_data_parallel"
            else (net.to(device="cuda") if use_gpu == "cuda" else net)
        )

        # train model using several optimization steps on Hessian data

        # create entire Hessian data as a batch
        hessian_dataset = (
            ExplicitDataset(train_samples, train_labels, use_gpu)
            if not unpack_inputs
            else UnpackDataset(train_samples, train_labels, use_gpu)
        )
        batch = next(iter(DataLoader(hessian_dataset, batch_size=num_train)))

        optimizer = torch.optim.Adam(net.parameters())
        num_steps = 200
        criterion = nn.MSELoss(reduction="sum")
        for _ in range(num_steps):
            optimizer.zero_grad()
            output = net_adjusted(*batch[:-1])
            loss = criterion(output, batch[-1])
            loss.backward()
            optimizer.step()

        net_adjusted = (
            _wrap_model_in_dataparallel(net) if use_gpu == "cuda_data_parallel" else net
        )

    training_data = (
        net_adjusted,
        dataset,
    )

    test_data = (
        _move_sample_to_cuda(test_samples)
        if isinstance(test_samples, list) and use_gpu
        else test_samples.cuda()
        if use_gpu
        else test_samples,
        test_labels.cuda() if use_gpu else test_labels,
    )

    return (*training_data, *test_data)


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


USE_GPU_LIST = (
    [False, "cuda"]
    if torch.cuda.is_available() and torch.cuda.device_count() != 0
    else [False]
) 
# in theory, can also include 'cuda_data_parallel, but all implementations of
# `EmbedderBase` do not support `DataParallel`.  TODO: support it.
