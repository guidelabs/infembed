from typing import Callable, Union
from unittest import TestCase
from infembed.embedder._core.arnoldi_embedder import ArnoldiEmbedder
from infembed.embedder._core.fast_kfac_embedder import FastKFACEmbedder
from infembed.embedder._core.gradient_embedder import GradientEmbedder
from infembed.embedder._core.kfac_embedder import KFACEmbedder
from infembed.embedder._utils.common import NotFitException, _format_inputs_dataset
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
        embedder.fit(train_dataloader)

        embeddings = embedder.fit(train_dataloader).predict(train_dataloader)

        # get total number of parameters where `requires_grad=True`
        num_requires_grad = int(
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        )

        # it should equal dimension of embedding
        assert (
            num_requires_grad == embeddings.shape[1]
        ), "embedding dim does not equal the number of parameters for which `requires_grad=True`"
