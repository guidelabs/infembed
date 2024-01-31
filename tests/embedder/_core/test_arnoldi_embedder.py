from typing import Callable, Union
from unittest import TestCase
from infembed.embedder._core.fast_kfac_embedder import FastKFACEmbedder
from infembed.embedder._core.kfac_embedder import KFACEmbedder
from infembed.embedder._core.arnoldi_embedder import ArnoldiEmbedder
from infembed.embedder._core.naive_embedder import NaiveEmbedder
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


class TestArnoldiEmbedder(TestCase):
    @parameterized.expand(
        [
            (
                embedder_constructor_1,
                embedder_constructor_2,
                delta,
                unpack_inputs,
                use_gpu,
            )
            for use_gpu in USE_GPU_LIST
            for (embedder_constructor_1, embedder_constructor_2, delta) in [
                # compare implementations, when considering only 1 layer
                (
                    EmbedderConstructor(
                        NaiveEmbedder,
                        layers=["linear1"],
                        projection_dim=5,
                        show_progress=False,
                        name="NaiveEmbedder_linear1",
                    ),
                    EmbedderConstructor(
                        ArnoldiEmbedder,
                        layers=["linear1"],
                        arnoldi_dim=50,
                        arnoldi_tol=1e-5,  # set low enough so that arnoldi subspace
                        # is large enough
                        projection_dim=5,
                        show_progress=False,
                        name="ArnoldiEmbedder_linear1",
                    ),
                    1e-2,
                ),
                # compare implementations, when considering all layers
                (
                    EmbedderConstructor(
                        NaiveEmbedder,
                        layers=None,
                        projection_dim=5,
                        show_progress=False,
                        name="NaiveEmbedder_all_layers",
                    ),
                    EmbedderConstructor(
                        ArnoldiEmbedder,
                        layers=None,
                        arnoldi_dim=50,
                        arnoldi_tol=1e-5,  # set low enough so that arnoldi subspace
                        # is large enough
                        projection_dim=5,
                        show_progress=False,
                        name="ArnoldiEmbedder_all_layers",
                    ),
                    1e-2,
                ),
            ]
            for unpack_inputs in [
                False,
                True,
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_compare_implementations_naive_vs_arnoldi(
        self,
        embedder_constructor_1: Callable,
        embedder_constructor_2: Callable,
        delta: float,
        unpack_inputs: bool,
        use_gpu: Union[bool, str],
    ):
        """
        this compares 2 embedder implementations on a trained 2-layer NN model.
        the implementations we compare are `NaiveEmbedder` and
        `ArnoldiEmbedder`. because the model is trained, calculations
        are more numerically stable, so that we can project to a higher dimension (5).
        """
        _test_compare_implementations(
            self,
            'trained_NN',
            embedder_constructor_1,
            embedder_constructor_2,
            delta,
            unpack_inputs,
            use_gpu,
        )