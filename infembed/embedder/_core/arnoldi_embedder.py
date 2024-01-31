from dataclasses import dataclass
import logging
from typing import Callable, List, Optional, Tuple, Union
from infembed.embedder._core.embedder_base import EmbedderBase
from infembed.embedder._utils.common import (
    NotFitException,
    _check_loss_fn,
    _compute_jacobian_sample_wise_grads_per_batch,
    _parameter_add,
    _parameter_dot,
    _parameter_linear_combination,
    _parameter_multiply,
    _parameter_to,
    _progress_bar_constructor,
    _set_active_parameters,
    _top_eigen,
)
from infembed.embedder._utils.gradient import _extract_parameters_from_layers
from infembed.embedder._utils.hvp import AutogradHVP
from torch.nn import Module
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import dill as pickle
import logging
from infembed.utils.common import profile


@dataclass
class ArnoldiEmbedderFitResults:
    """
    stores the results of calling `ArnoldiEmbedder.fit`
    """

    R: List[Tuple[Tensor, ...]]


def _parameter_arnoldi(
    hvp: Callable,
    b: Tuple[Tensor, ...],
    n: int,
    tol: float,
    projection_device: torch.device,
    show_progress: bool,
) -> Tuple[List[Tuple[Tensor, ...]], Tensor]:
    r"""
    Given `hvp`, a function which computes the Hessian-vector product of an arbitrary
    vector `v` with an implicitly-defined Hessian matrix `A`, performs the Arnoldi
    iteration for `A` for `n` iterations.  (We use `A`, not `H` to refer to the
    Hessian, unlike elsewhere, because `H` is already used in the below explanation
    of the Arnoldi iteration.)

    For more details on the Arnoldi iteration, please see Trefethen and Bau, Chp 33.
    Running Arnoldi iteration for n iterations gives a basis for the Krylov subspace
    spanned by :math`\{b, Ab,..., A^{n-1}b\}`, as well as a `n+1` by `n` matrix
    :math`H_n` which is upper Hessenberg (all entries below the diagonal, except those
    adjoining it, are 0), whose first n rows represent the restriction of `A` to the
    Krylov subspace, using the basis. Here, `b` is an arbitrary initialization basis
    vector. The basis is assembled into a `D` by `n+1` matrix, where the last
    column is a "correction factor", i.e. not part of the basis, denoted
    :math`Q_{n+1}`. Letting :math`Q_n` denote the matrix with the first n columns of
    :math`Q_{n+1}`, the following equality is satisfied: :math`A=Q_{n+1} H_n Q_n'`.

    In this implementation, `v` is not actually a vector, but instead a tuple of
    tensors, because `hvp` being a Hessian-vector product, `v` lies in parameter-space,
    which Pytorch represents as tuples of tensors. This implementation avoids
    flattening `v` to a 1D tensor, which leads to scalability gains.

    Args:
        hvp (Callable): A callable that accepts an arbitrary tuple of tensors
                `v`, which represents a parameter, and returns
                `Av`, i.e. the multiplication of `v` with an implicitly defined matrix
                `A` of compatible dimension, which in practice is a Hessian-vector
                product.
        b (tensor): The Arnoldi iteration requires an initialization basis to
                construct the basis, typically randomly chosen. This is that basis,
                and is a tuple of tensors. We assume that the device of `b` is the same
                as the required device of input `v` to `hvp`. For example, if `hvp`
                computes HVP using a model that is on the GPU, then `b` should also be
                on the GPU.
        n (int): The number of iterations to run the iteration for.
        tol (float, optional): After many iterations, the already-obtained
                basis vectors may already approximately span the Krylov subspace,
                in which case the addition of additional basis vectors involves
                normalizing a vector with a small norm. These vectors are not
                necessary to include in the basis and furthermore, their small norm
                leads to numerical issues. Therefore we stop the Arnoldi iteration
                when the addition of additional vectors involves normalizing a
                vector with norm below a certain threshold. This argument specifies
                that threshold.
                Default: 1e-4
        projection_device (torch.device) The returned quantities (which will be used
                to define a projection of parameter-gradients, hence the name) are
                potentially memory intensive, because they represent a basis of a
                subspace in the space of parameters, which are potentially
                high-dimensional. Therefore we need to be careful of out-of-memory
                GPU errors. This argument represents the device where the returned
                quantities should be stored, and its choice requires balancing
                speed with GPU memory.
        show_progress (bool): If true, the progress of the iteration (i.e. number of
                basis vectors already determined) will be displayed. It will try to
                use tqdm if available for advanced features (e.g. time estimation).
                Otherwise, it will fallback to a simple output of progress.

    Returns:
        qs (list of tuple of tensors): A list of tuple of tensors, whose first `n`
                elements contain a basis for the Krylov subspace.
        H (tensor): A tensor with shape `(n+1, n)` whose first `n` rows represent
                the restriction of `A` to the Krylov subspace.
    """
    # because the HVP is the computational bottleneck, we always do HVP on
    # the same device as the model, which is assumed to be the device `b` is on
    computation_device = next(iter(b)).device

    # all entries of `b` have the same dtype, and so can be used to determine dtype
    # of `H`
    H = torch.zeros(n + 1, n, dtype=next(iter(b)).dtype).to(device=projection_device)
    qs = [
        _parameter_to(
            _parameter_multiply(b, 1.0 / _parameter_dot(b, b) ** 0.5),
            device=projection_device,
        )
    ]

    iterates = range(1, n + 1)
    if show_progress:
        iterates = tqdm(iterates, desc="Running Arnoldi Iteration for step")

    for k in iterates:
        logging.info(f"arnoldi iteration step {k}")
        v = _parameter_to(
            hvp(_parameter_to(qs[k - 1], device=computation_device)),
            device=projection_device,
        )

        for i in range(k):
            H[i, k - 1] = _parameter_dot(qs[i], v)
            v = _parameter_add(v, _parameter_multiply(qs[i], -H[i, k - 1]))
        H[k, k - 1] = _parameter_dot(v, v) ** 0.5
        #logging.info(f"tol, {H[k, k - 1]}")
        if H[k, k - 1] < tol:
            break
        qs.append(_parameter_multiply(v, 1.0 / H[k, k - 1]))

    return qs[:k], H[:k, : k - 1]


def _parameter_distill(
    qs: List[Tuple[Tensor, ...]],
    H: Tensor,
    k: Optional[int],
    hessian_reg: float,
    hessian_inverse_tol: float,
):
    """
    This takes the output of `_parameter_arnoldi`, and extracts the top-k eigenvalues
    / eigenvectors of the matrix that `_parameter_arnoldi` found the Krylov subspace
    for. In this documentation, we will refer to that matrix by `A`.

    Args:
        qs (list of tuple of tensors): A list of tuple of tensors, whose first `N`
                elements contain a basis for the Krylov subspace.
        H (tensor): A tensor with shape `(N+1, N)` whose first `N` rows represent
                the restriction of `A` to the Krylov subspace.
        k (int): The number of top eigenvalues / eigenvectors to return. Note that the
                actual number returned may be less, due to filtering based on
                `hessian_inverse_tol`.
        hessian_reg (float): hessian_reg (float): We add an entry to the diagonal of
                `H` to encourage it to be positive definite. This is that entry.
        hessian_inverse_tol (float): To compute the "square root" of `H` using the top
                eigenvectors / eigenvalues, the eigenvalues should be positive, and
                furthermore if above a tolerance, the inversion will be more
                numerically stable. Therefore, we only return eigenvectors /
                eigenvalues where the eigenvalue is above a tolerance. This argument
                specifies that tolerance. We do not compute the square root in this
                function, but assume the output of this function will be used for
                computing it, hence the need for this argument.

    Returns:
        (eigenvalues, eigenvectors) (tensor, list of tuple of tensors): `eigenvalues`
                is a 1D tensor of the top eigenvalues of `A`. Note that due to
                filtering based on `hessian_inverse_tol`, the actual number of
                eigenvalues may be less than `k`. The eigenvalues are in ascending
                order, mimicking the convention of `torch.linalg.eigh`. `eigenvectors`
                are the corresponding eigenvectors. Since `A` represents the Hessian
                of parameters, with the parameters represented as a tuple of tensors,
                the eigenvectors, because they represent parameters, are also
                tuples of tensors. Therefore, `eigenvectors` is a list of tuple of
                tensors.
    """
    # get rid of last basis of qs, last column of H, since they are not part of
    # the decomposition
    qs = qs[:-1]
    H = H[:-1]

    # if arnoldi basis is empty, raise exception
    if len(qs) == 0:
        raise Exception(
            "Arnoldi basis is empty. Consider increasing the `arnoldi_tol` argument"
        )

    # ls, vs are the top eigenvalues / eigenvectors.  however, the eigenvectors are
    # expressed as coordinates using the Krylov subspace basis, qs (each column of vs
    # represents a different eigenvector).
    ls, vs = _top_eigen(H, k, hessian_reg, hessian_inverse_tol)

    # if no positive eigenvalues exist, we cannot compute a low-rank
    # approximation of the square root of the hessian H, so raise exception
    if vs.shape[1] == 0:
        raise Exception(
            "Restriction of Hessian to Krylov subspace has no positive "
            "eigenvalues, so cannot take its square root."
        )

    # we want to express the top eigenvectors as coordinates using the standard basis.
    # each column of vs represents a different eigenvector, expressed as coordinates
    # using the Krylov subspace basis.  to express the eigenvector using the standard
    # basis, we use it as the coefficients in a linear combination with the Krylov
    # subspace basis, qs.
    vs_standard = [_parameter_linear_combination(qs, v) for v in vs.T]

    return ls, vs_standard


class ArnoldiEmbedder(EmbedderBase):
    """
    Computes embeddings which are "influence embeddings" - vectors such that the
    dot-product of two examples' embeddings is the "influence" of one example on the
    other, where the general notion of influence is as defined in Koh and Liang
    (https://arxiv.org/abs/1703.04730).  See the paper by Wang and Adebayo et al
    (https://arxiv.org/abs/2312.04712) for more background on influence embeddings.

    Influence embeddings are dependent on the exact definition and implementation of
    influence that is used.  This implementation is based on an implementation of
    influence (see Schioppa et al, https://arxiv.org/abs/2112.03052) that uses the
    Arnoldi iteration to approximate the Hessian without explicitly forming the actual
    Hessian.
    """
    def __init__(
        self,
        model: Module,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[Union[Module, Callable]] = None,
        test_loss_fn: Optional[Union[Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        projection_dim: Union[int, List[int]] = 50,
        seed: int = 0,
        arnoldi_dim: int = 200,
        arnoldi_tol: float = 1e-1,
        hessian_reg: float = 1e-6,
        hessian_inverse_tol: float = 1e-5,
        projection_on_cpu: bool = True,
        show_progress: bool = False,
    ):
        """
        Args:
            model (Module): The model used to compute the embeddings.
            layers (list of str, optional): names of modules in which to consider
                    gradients.  If `None` or not provided, all modules will be used.
                    There is a caveat: `KFACEmbedder` can only consider gradients
                    in layers which are `Linear` or `Conv2d`.  Thus regardless of
                    the modules specified by `layers`, only layers in them which are
                    of those types will be used for calculating gradients.  The modules
                    should not be nested, i.e. if one module is specified, do not
                    also specify its submodules.
                    Default: `None`
            loss_fn (Module or Callable, optional): The loss function used to compute the
                    Hessian.  It should behave like a "reduction" loss function, where
                    reduction is either 'sum', 'mean', or 'none', and have a
                    `reduction` attribute.  For example, `BCELoss(reduction='sum')`
                    could be a valid loss function.  See the caveat under the
                    description for the `sample_wise_grads_per_batch` argument.  If None,
                    the loss is the output of `model`, which is assumed to be a single
                    scalar for a batch.
                    Default: None
            test_loss_fn: (Module or callable, optional): The loss function used to compute
                    the 'influence explanations'.  This argument should not matter for
                    most use cases.  If None, is assumed to be the same as `loss_fn`.
            sample_wise_grads_per_batch (bool, optional): Whether to use an efficiency
                    trick to compute the per-example gradients.  If True, `loss_fn` must
                    behave like a `reduction='sum'` or `reduction='sum'` loss function,
                    i.e. `BCELoss(reduction='sum')` or `BCELoss(reduction='mean')`.  If
                    False, `loss_fn` must behave like a `reduction='none'` loss
                    function, i.e. `BCELoss(reduction='none')`.
                    Default: True
            projection_dim (int, optional): This implementation produces a low-rank
                    approximation of the (inverse) Hessian. This is the rank of that
                    approximation, and also corresponds to the dimension of the
                    embeddings that are computed.
                    Default: 50
            seed (int, optional): Random seed for reproducibility.
                    Default: 0
            arnoldi_dim (int, optional): Calculating the low-rank approximation of the
                    (inverse) Hessian requires approximating the Hessian's top
                    eigenvectors / eigenvalues. This is done by first computing a
                    Krylov subspace via the Arnoldi iteration, and then finding the top
                    eigenvectors / eigenvalues of the restriction of the Hessian to the
                    Krylov subspace. Because only the top eigenvectors / eigenvalues
                    computed in the restriction will be similar to those in the full
                    space, `arnoldi_dim` should be chosen to be larger than
                    `projection_dim`. In the paper, they often choose `projection_dim`
                    to be between 10 and 100, and `arnoldi_dim` to be 200. Please see
                    the paper as well as Trefethen and Bau, Chapters 33-34 for more
                    details on the Arnoldi iteration.
                    Default: 200
            arnoldi_tol (float, optional): After many iterations, the already-obtained
                    basis vectors may already approximately span the Krylov subspace,
                    in which case the addition of additional basis vectors involves
                    normalizing a vector with a small norm. These vectors are not
                    necessary to include in the basis and furthermore, their small norm
                    leads to numerical issues. Therefore we stop the Arnoldi iteration
                    when the addition of additional vectors involves normalizing a
                    vector with norm below a certain threshold. This argument specifies
                    that threshold.
                    Default: 1e-4
            hessian_reg (float, optional): This implementation computes the eigenvalues /
                    eigenvectors of Hessians.  We add an entry to the Hessian's
                    diagonal entries before computing them.  This is that entry.
                    Default: 1e-6
            hessian_inverse_tol (float): This implementation computes the
                    pseudo-inverse of the (square root of) Hessians.  This is the
                    tolerance to use in that computation.
                    Default: 1e-6
            projection_on_cpu (bool, optional): Whether to move the projection,
                    i.e. low-rank approximation of the inverse Hessian, to cpu, to save
                    gpu memory.
                    Default: True
            show_progress (bool, optional): Whether to show the progress of
                    computations in both the `fit` and `predict` methods.
                    Default: False
        """
        self.model = model

        self.loss_fn = loss_fn
        # If test_loss_fn not provided, it's assumed to be same as loss_fn
        self.test_loss_fn = loss_fn if test_loss_fn is None else test_loss_fn
        self.sample_wise_grads_per_batch = sample_wise_grads_per_batch

        # we save the reduction type for both `loss_fn` and `test_loss_fn` because
        # 1) if `sample_wise_grads_per_batch` is true, the reduction type is needed
        # to compute per-example gradients, and 2) regardless, reduction type for
        # `loss_fn` is needed to compute the Hessian.

        # check `loss_fn`
        self.reduction_type = _check_loss_fn(
            loss_fn, "loss_fn", sample_wise_grads_per_batch
        )
        # check `test_loss_fn` if it was provided
        if not (test_loss_fn is None):
            self.test_reduction_type = _check_loss_fn(
                test_loss_fn, "test_loss_fn", sample_wise_grads_per_batch
            )
        else:
            self.test_reduction_type = self.reduction_type

        self.layer_modules = None
        if not (layers is None):
            # TODO: should let `self.layer_modules` only contain supported layers
            self.layer_modules = _set_active_parameters(model, layers)
        else:
            # only use supported layers.  TODO: add warning that some layers are not supported
            self.layer_modules = list(model.modules())

        # below initializations are specific to `ArnoldiEmbedder`
        self.projection_dim = projection_dim

        torch.manual_seed(seed)  # for reproducibility

        self.arnoldi_dim = arnoldi_dim
        self.arnoldi_tol = arnoldi_tol
        self.hessian_reg = hessian_reg
        self.hessian_inverse_tol = hessian_inverse_tol

        # infer the device the model is on.  all parameters are assumed to be on the
        # same device
        self.model_device = next(model.parameters()).device

        self.projection_on_cpu = projection_on_cpu
        self.show_progress = show_progress

    def fit(
        self,
        dataloader: DataLoader,
    ):
        r"""
        Does the computation needed for computing embeddings, which is
        finding the top eigenvectors / eigenvalues of the Hessian, computed
        using `dataloader`.

        Args:
            dataloader (DataLoader): The dataloader containing data needed to learn how
                    to compute the embeddings
        """
        logging.info("start arnoldi iteration")
        self.fit_results = self._retrieve_projections_arnoldi_embedder(
            dataloader, self.projection_on_cpu, self.show_progress
        )
        return self

    def _retrieve_projections_arnoldi_embedder(
        self,
        dataloader: DataLoader,
        projection_on_cpu: bool,
        show_progress: bool,
    ) -> ArnoldiEmbedderFitResults:
        """

        Returns the `R` described in the documentation for
        `ArnoldiEmbedder`. The returned `R` represents a set of
        parameters in parameter space. However, since this implementation does *not*
        flatten parameters, each of those parameters is represented as a tuple of
        tensors. Therefore, `R` is represented as a list of tuple of tensors, and
        can be viewed as a linear function that takes in a tuple of tensors
        (representing a parameter), and returns a vector, where the i-th entry is
        the dot-product (as it would be defined over tuple of tensors) of the parameter
        (i.e. the input to the linear function) with the i-th entry of `R`.

        Can specify that projection should always be saved on cpu. if so, gradients are
        always moved to same device as projections before multiplying (moving
        projections to gpu when multiplying would defeat the purpose of moving them to
        cpu to save gpu memory).

        Returns a `dataclass` with the following attributes:
            R (list of tuple of tensors): List of tuple of tensors of length
                    `projection_dim` (initialization argument). Each element
                    corresponds to a parameter in parameter-space, is represented as a
                    tuple of tensors, and together, define a projection that can be
                    applied to parameters (represented as tuple of tensors).
        """
        # create function that computes hessian-vector product, given a vector
        # represented as a tuple of tensors

        HVP = AutogradHVP(show_progress)

        HVP.setup(
            self.model,
            dataloader,
            self.layer_modules,
            self.reduction_type,
            self.loss_fn,
        )

        # now that can compute the hessian-vector product (of loss over `dataloader`),
        # can perform arnoldi iteration

        # we always perform the HVP computations on the device where the model is.
        # effectively this means we do the computations on gpu if available. this
        # is necessary because the HVP is computationally expensive.

        # get initial random vector, and place it on the same device as the model.
        # `_parameter_arnoldi` needs to know which device the model is on, and
        # will infer it through the device of this random vector
        # the order of parameters is determined using the same logic as in `HVP`, based
        # on `self.layer_modules`
        params = (
            self.model.parameters()
            if self.layer_modules is None
            else _extract_parameters_from_layers(self.layer_modules)
        )
        b = _parameter_to(
            tuple(torch.randn_like(param) for param in params),
            device=self.model_device,
        )

        # perform the arnoldi iteration, see its documentation for what its return
        # values are.  note that `H` is *not* the Hessian.
        logging.info("start `_parameter_arnoldi`")
        qs, H = _parameter_arnoldi(
            HVP,
            b,
            self.arnoldi_dim,
            self.arnoldi_tol,
            torch.device("cpu") if projection_on_cpu else self.model_device,
            show_progress,
        )

        # `ls`` and `vs`` are (approximately) the top eigenvalues / eigenvectors of the
        # matrix used (implicitly) to compute Hessian-vector products by the `HVP`
        # input to `_parameter_arnoldi`. this matrix is the Hessian of the loss,
        # summed over the examples in `dataloader`. note that because the vectors in
        # the Hessian-vector product are actually tuples of tensors representing
        # parameters, `vs`` is a list of tuples of tensors.  note that here, `H` is
        # *not* the Hessian (`qs` and `H` together define the Krylov subspace of the
        # Hessian)

        logging.info("start `_parameter_distill`")
        ls, vs = _parameter_distill(
            qs, H, self.projection_dim, self.hessian_reg, self.hessian_inverse_tol
        )

        # if `vs` were a 2D tensor whose columns contain the top eigenvectors of the
        # aforementioned hessian, then `R` would be `vs @ torch.diag(ls ** -0.5)`, i.e.
        # scaling each column of `vs` by the corresponding entry in `ls ** -0.5`.
        # however, since `vs` is instead a list of tuple of tensors, `R` should be
        # a list of tuple of tensors, where each entry in the list is scaled by the
        # corresponding entry in `ls ** 0.5`, which we first compute.
        ls = (1.0 / ls) ** 0.5

        # then, scale each entry in `vs` by the corresponding entry in `ls ** 0.5`
        # since each entry in `vs` is a tuple of tensors, we use a helper function
        # that takes in a tuple of tensors, and a scalar, and multiplies every tensor
        # by the scalar.
        return ArnoldiEmbedderFitResults(
            [_parameter_multiply(v, l) for (v, l) in zip(vs, ls)]
        )

    def predict(self, dataloader: DataLoader) -> Tensor:
        """
        Returns the embeddings for `dataloader`.

        Args:
            dataloader (`DataLoader`): dataloader whose examples to compute embeddings
                    for.
        """
        if self.fit_results is None:
            raise NotFitException(
                "The results needed to compute embeddings not available.  Please either call the `fit` or `load` methods."
            )

        if self.show_progress:
            dataloader = _progress_bar_constructor(
                self, dataloader, "embeddings", "training data"
            )

        # always return embeddings on cpu
        return_device = torch.device("cpu")

        # choose the correct loss function and reduction type based on `test`
        # actually, `test` is always true
        test = True
        loss_fn = self.test_loss_fn if test else self.loss_fn
        reduction_type = self.test_reduction_type if test else self.reduction_type

        # define a helper function that returns the embeddings for a batch
        def get_batch_embeddings(batch):
            # get gradient
            features, labels = tuple(batch[0:-1]), batch[-1]
            # `jacobians`` is a tensor of tuples. unlike parameters, however, the first
            # dimension is a batch dimension
            jacobians = _compute_jacobian_sample_wise_grads_per_batch(
                self, features, labels, loss_fn, reduction_type
            )

            # `jacobians`` contains the per-example parameters for a batch. this
            # function takes in `params`, a tuple of tensors representing a single
            # parameter setting, and for each example, computes the dot-product of its
            # per-example parameter with `params`. in other words, given `params`,
            # representing a basis vector, this function returns the coordinate of
            # each example in the batch along that basis. note that `jacobians` and
            # `params` are both tuple of tensors, with the same length. however, a
            # tensor in `jacobians` always has dimension 1 greater than the
            # corresponding tensor in `params`, because the tensors in `jacobians` have
            # a batch dimension (the 1st). to do this computation, the naive way would
            # be to convert `jacobians` to a list of tuple of tensors, and use
            # `_parameter_dot` to take the dot-product of each element in the list
            # with `params` to get a 1D tensor whose length is the batch size. however,
            # we can do the same computation without actually creating that list of
            # tuple of tensors by using broadcasting.
            def get_batch_coordinate(params):
                batch_coordinate = 0
                for _jacobians, param in zip(jacobians, params):
                    batch_coordinate += torch.sum(
                        _jacobians * param.to(device=self.model_device).unsqueeze(0),
                        dim=tuple(range(1, len(_jacobians.shape))),
                    )
                return batch_coordinate.to(device=return_device)

            # to get the embedding for the batch, we get the coordinates for the batch
            # corresponding to one parameter in `R`. We do this for every parameter in
            # `R`, and then concatenate.
            return torch.stack(
                [get_batch_coordinate(params) for params in self.fit_results.R],
                dim=1,
            )

        with torch.no_grad():
            logging.info("compute embeddings")
            return torch.cat(
                [get_batch_embeddings(batch) for batch in dataloader], dim=0
            )

    def save(self, path: str):
        """
        This method saves the results of `fit` to a file.

        Args:
            path (str): path of file to save results in.
        """
        with open(path, "wb") as f:
            pickle.dump(self.fit_results, f)

    def load(self, path: str):
        """
        Loads the results saved by the `save` method.  Instead of calling `fit`, one
        can instead call `load`.

        Args:
            path (str): path of file to load results from.
        """
        with open(path, "rb") as f:
            self.fit_results = pickle.load(f)

    def reset(self):
        """
        Removes the effect of calling `fit` or `load`
        """
        self.fit_results = None