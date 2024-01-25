from typing import Callable, Union
from unittest import TestCase
from infembed.embedder._core.fast_kfac_embedder import FastKFACEmbedder
from infembed.embedder._core.kfac_embedder import KFACEmbedder
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
import torch


class TestFastKFACEmbedder(TestCase):
    @parameterized.expand(
        [
            (
                embedder_constructor_1,
                embedder_constructor_2,
                delta,
                unpack_inputs,
                use_gpu,
                model_type,
            )
            for use_gpu in USE_GPU_LIST
            for (
                embedder_constructor_1,
                embedder_constructor_2,
                delta,
                model_type,
            ) in [
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear"],
                        layer_block_projection_dim=None,
                        hessian_inverse_tol=0.0,
                        projection_dim=None,
                    ),
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear"],
                        layer_projection_dim=None,
                        independent_factors=True,
                        hessian_inverse_tol=0.0,
                    ),
                    1e-2,
                    "one_layer_linear",
                ),
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear1", "linear2"],
                        layer_block_projection_dim=None,
                        hessian_inverse_tol=0.0,
                        projection_dim=None,
                    ),
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear1", "linear2"],
                        layer_projection_dim=None,
                        independent_factors=True,
                        hessian_inverse_tol=0.0,
                    ),
                    1e-2,
                    "seq",
                ),
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        layer_block_projection_dim=100,
                        # hessian_inverse_tol=0.0,
                        hessian_inverse_tol=-1e-2,
                        hessian_reg=1e-8,
                        projection_dim=None,
                    ),
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        layer_projection_dim=100,
                        independent_factors=True,
                        # hessian_inverse_tol=0.0,
                        hessian_inverse_tol=-1e-2,
                        hessian_reg=1e-8,
                    ),
                    5e-0,
                    "conv",
                ),
            ]
            for unpack_inputs in [
                False,
                True,
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_compare_implementations_KFAC_vs_FastKFAC(
        self,
        embedder_constructor_1: Callable,
        embedder_constructor_2: Callable,
        delta: float,
        unpack_inputs: bool,
        use_gpu: Union[bool, str],
        model_type: str,
    ):
        """
        this compares `KFACEmbedder` with `FastKFACEmbedder` where
        `projection_dim=None`.  in this setting, up to numerical issues, influence
        should be equal for the two implementations.  of course, we do not know if
        `KFACEmbedder` is actually correct, so a TODO is to check that in a test.
        """
        _test_compare_implementations(
            self,
            model_type,
            embedder_constructor_1,
            embedder_constructor_2,
            delta,
            unpack_inputs,
            use_gpu,
        )

    @parameterized.expand(
        [
            (
                embedder_constructor,
                model_type,
                projection_dim,
            )
            for (
                embedder_constructor,
                model_type,
                projection_dim,
            ) in [
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear"],
                        hessian_inverse_tol=0.0,
                    ),
                    "one_layer_linear",
                    5,  # this model only has 5 parameters, so this is edge case
                ),
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear1", "linear2"],
                        hessian_inverse_tol=0.0,
                    ),
                    "seq",
                    15,
                ),
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        # hessian_inverse_tol=0.0,
                        hessian_inverse_tol=0.0,
                        hessian_reg=1e-8,
                    ),
                    "conv",
                    15,
                ),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_correct_dimensions(
        self,
        embedder_constructor,
        model_type: str,
        projection_dim: int,
    ):
        """
        tests that when `projection_dim` is specified for `FastKFACEmbedder`, the
        projections are of the specified dimension.  set `hessian_inverse_tol` to 0 so
        that don't lose any dimensions because eigenvalues are too close to 0.  since
        using KFAC, all eigenvalues should be positive.
        """
        (
            net,
            train_dataset,
            _,
            _,
        ) = get_random_model_and_data(
            unpack_inputs=False,
            use_gpu=False,
            model_type=model_type,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=5)

        criterion = nn.MSELoss(reduction="none")

        embedder = embedder_constructor(
            model=net,
            loss_fn=criterion,
            projection_dim=projection_dim,
        )

        embeddings = embedder.fit(train_dataloader).predict(train_dataloader)
        assert embeddings.shape[1] == projection_dim