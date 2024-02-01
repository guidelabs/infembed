from typing import Callable, List, Optional, Union
from unittest import TestCase
from infembed.embedder._core.arnoldi_embedder import ArnoldiEmbedder
from infembed.embedder._core.fast_kfac_embedder import FastKFACEmbedder
from infembed.embedder._core.gradient_embedder import (
    GradientEmbedder,
    PCAGradientEmbedder,
)
from infembed.embedder._core.kfac_embedder import KFACEmbedder
from infembed.embedder._utils.common import NotFitException, _format_inputs_dataset
from infembed.embedder._utils.gradient import _extract_parameters_from_layers
from .._utils.common import (
    EmbedderConstructor,
    _test_compare_implementations,
    USE_GPU_LIST,
    get_random_model_and_data,
)
from parameterized import parameterized
from ...utils.common import assertTensorAlmostEqual, build_test_name_func
from torch.utils.data import DataLoader
import torch.nn as nn
import tempfile


class TestEmbeddingDim(TestCase):
    @parameterized.expand(
        [
            (
                embedder_constructor,
                model_type,
                reduction,
                sample_wise_grads_per_batch,
            )
            for (reduction, sample_wise_grads_per_batch) in [
                ("none", False),
                ("sum", True),
            ]
            for (
                embedder_constructor,
                model_type,
            ) in [
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear1"],
                        layer_block_projection_dim=None,
                        hessian_inverse_tol=0.0,
                        projection_dim=None,
                    ),
                    "seq",
                ),
                (
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear1"],
                        layer_projection_dim=None,
                        independent_factors=True,
                        hessian_inverse_tol=0.0,
                    ),
                    "seq",
                ),
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear1", "linear2"],
                        layer_block_projection_dim=None,
                        hessian_inverse_tol=0.0,
                        projection_dim=None,
                    ),
                    "seq",
                ),
                (
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear1", "linear2"],
                        layer_projection_dim=None,
                        independent_factors=True,
                        hessian_inverse_tol=0.0,
                    ),
                    "seq",
                ),
                (
                    EmbedderConstructor(
                        GradientEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        # projection_dim=100,
                        # hessian_inverse_tol=0.0,
                        # hessian_inverse_tol=-1e-2,
                        # hessian_reg=1e-8,
                    ),
                    "conv",
                ),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_embedding_dim(
        self,
        embedder_constructor: Callable,
        model_type: str,
        reduction: str,
        sample_wise_grads_per_batch: bool,
    ):
        """
        for implementations where it's possible to specify that no dimension reduction
        be done, checks that in that scenario, the dimension of embedding is equal to
        the number of parameters for which `requires_grad=True`.
        """
        (
            net,
            train_dataset,
            _,
            _,
        ) = get_random_model_and_data(
            unpack_inputs=False,
            model_type=model_type,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=5)
        criterion = nn.MSELoss(reduction=reduction)
        embedder = embedder_constructor(
            model=net,
            loss_fn=criterion,
            sample_wise_grads_per_batch=sample_wise_grads_per_batch,
        )

        embeddings = embedder.fit(train_dataloader).predict(train_dataloader)

        # get total number of parameters where `requires_grad=True`
        num_requires_grad = int(
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        )

        # it should equal dimension of embedding
        assert (
            num_requires_grad == embeddings.shape[1]
        ), "embedding dim does not equal the number of parameters for which `requires_grad=True`"

    @parameterized.expand(
        [
            (
                embedder_constructor,
                layers,
                model_type,
                reduction,
                sample_wise_grads_per_batch,
            )
            for (
                layers,
                model_type,
            ) in [
                (["linear1.linear1", "linear2.linear2"], "two_layer_with_submodule"),
                (None, "two_layer_with_submodule"),
            ]
            for (reduction, sample_wise_grads_per_batch) in [
                ("sum", True),
                ("none", False),
            ]
            for embedder_constructor in [
                EmbedderConstructor(
                    FastKFACEmbedder,
                    layer_block_projection_dim=None,
                    hessian_inverse_tol=0.0,
                    projection_dim=None,
                ),
                EmbedderConstructor(
                    KFACEmbedder,
                    layer_projection_dim=None,
                    independent_factors=True,
                    hessian_inverse_tol=0.0,
                ),
                EmbedderConstructor(
                    ArnoldiEmbedder,
                ),
                EmbedderConstructor(
                    GradientEmbedder,
                ),
                # EmbedderConstructor(
                #     PCAGradientEmbedder,
                # ),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_active_parameters(
        self,
        embedder_constructor: Callable,
        layers: Optional[List],
        model_type: str,
        reduction: str,
        sample_wise_grads_per_batch: bool,
    ):
        """
        tests that when computing gradients, no parameter is represented more than
        once.  this could happen if one of the layers passed in is a submodul of
        another layer passed in.
        """
        (
            net,
            train_dataset,
            _,
            _,
        ) = get_random_model_and_data(
            unpack_inputs=False,
            model_type=model_type,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=5)
        criterion = nn.MSELoss(reduction=reduction)
        embedder = embedder_constructor(
            model=net,
            loss_fn=criterion,
            sample_wise_grads_per_batch=sample_wise_grads_per_batch,
            layers=layers,
        )

        # parameters are computed via a call to `_extract_parameters_from_layers`
        # so check that the parameters it returns has no duplicates
        layer_parameters = _extract_parameters_from_layers(
            embedder.layer_modules, check_duplicates=False
        )
        assert len(set(layer_parameters)) == len(
            layer_parameters
        ), "There are duplicate parameters in which gradients are considered."

        # then compute embeddings to make sure we can still compute them
        embeddings = embedder.fit(train_dataloader).predict(train_dataloader)
        assert embeddings.shape[1] > 0
