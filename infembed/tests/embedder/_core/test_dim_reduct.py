from typing import Callable, Union
from unittest import TestCase
from infembed.embedder._core.arnoldi_embedder import ArnoldiEmbedder
from infembed.embedder._core.fast_kfac_embedder import FastKFACEmbedder
from infembed.embedder._core.gradient_embedder import (
    GradientEmbedder,
    PCAGradientEmbedder,
)
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


class TestDimReduct(TestCase):
    @parameterized.expand(
        [
            (
                embedder_constructor,
                model_type,
                projection_dim,
            )
            for projection_dim in [
                2,
                5,
                10,
            ]
            for (
                embedder_constructor,
                model_type,
            ) in [
                (
                    EmbedderConstructor(
                        PCAGradientEmbedder,
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
    def test_pca_embedder(
        self,
        embedder_constructor: Callable,
        model_type: str,
        projection_dim: int,
    ):
        """
        tests that implementations of `PCAEmbedder` returns embeddings of the desired
        dimension, and that it works regardless of whether the desired dimension is
        larger or smaller than the batch size.
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
        criterion = nn.MSELoss(reduction="sum")
        embedder = embedder_constructor(
            model=net,
            loss_fn=criterion,
            projection_dim=projection_dim,
            sample_wise_grads_per_batch=True,
        )
        embedder.fit(train_dataloader)

        embeddings = embedder.fit(train_dataloader).predict(train_dataloader)

        assert embeddings.shape[1] == projection_dim, 'embeddings not of right dimension'