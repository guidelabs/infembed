from typing import Callable, Union
from unittest import TestCase
from infembed.embedder._core.fast_kfac_embedder import FastKFACEmbedder
from infembed.embedder._core.kfac_embedder import KFACEmbedder
from infembed.tests.embedder._utils.common import (
    EmbedderConstructor,
    _test_compare_implementations,
    build_test_name_func,
    USE_GPU_LIST,
)
from parameterized import parameterized


class TestFastKFACEmbedder(TestCase):
    @parameterized.expand(
        [
            (
                influence_constructor_1,
                influence_constructor_2,
                delta,
                unpack_inputs,
                use_gpu,
                model_type,
            )
            for use_gpu in USE_GPU_LIST
            for (
                influence_constructor_1,
                influence_constructor_2,
                delta,
                model_type,
            ) in [
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear"],
                        projection_dim=None,
                        hessian_inverse_tol=0.0,
                    ),
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear"],
                        projection_dim=None,
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
                        projection_dim=None,
                        hessian_inverse_tol=0.0,
                    ),
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear1", "linear2"],
                        projection_dim=None,
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
                        projection_dim=100,
                        # hessian_inverse_tol=0.0,
                        hessian_inverse_tol=-1e-2,
                        hessian_reg=1e-8,
                    ),
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        projection_dim=100,
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
        this compares `KFACEmbedder` with `FastKFACEmbeddern` where
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
