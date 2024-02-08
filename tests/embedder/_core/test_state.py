from typing import Callable, Optional, Union
from unittest import TestCase
from infembed.embedder._core.arnoldi_embedder import ArnoldiEmbedder
from infembed.embedder._core.fast_kfac_embedder import FastKFACEmbedder
from infembed.embedder._core.gradient_embedder import (
    GradientEmbedder,
    PCAGradientEmbedder,
)
from infembed.embedder._core.kfac_embedder import KFACEmbedder
from infembed.embedder._utils.common import NotFitException, _format_inputs_dataset
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
import tempfile


class TestSaveLoad(TestCase):
    @parameterized.expand(
        [
            (
                embedder_constructor,
                unpack_inputs,
                use_gpu,
                model_type,
                init_projection_on_cpu,
                load_projection_on_cpu,
                reduction,
                sample_wise_grads_per_batch,
            )
            for use_gpu in USE_GPU_LIST
            for (
                embedder_constructor,
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
                    "one_layer_linear",
                ),
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear"],
                        hessian_inverse_tol=0.0,
                        projection_dim=50,
                    ),
                    "one_layer_linear",
                ),
                (
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear"],
                        layer_projection_dim=None,
                        independent_factors=True,
                        hessian_inverse_tol=0.0,
                    ),
                    "one_layer_linear",
                ),
                (
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear"],
                        layer_projection_dim=None,
                        independent_factors=False,
                        hessian_inverse_tol=0.0,
                    ),
                    "one_layer_linear",
                ),
                (
                    EmbedderConstructor(
                        ArnoldiEmbedder,
                        layers=["linear"],
                        projection_dim=None,
                        hessian_inverse_tol=0.0,
                    ),
                    "one_layer_linear",
                ),
                (
                    EmbedderConstructor(
                        NaiveEmbedder,
                        layers=["linear"],
                        projection_dim=None,
                        hessian_inverse_tol=0.0,
                    ),
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
                    "seq",
                ),
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear1", "linear2"],
                        hessian_inverse_tol=0.0,
                        projection_dim=50,
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
                        KFACEmbedder,
                        layers=["linear1", "linear2"],
                        layer_projection_dim=None,
                        independent_factors=False,
                        hessian_inverse_tol=0.0,
                    ),
                    "seq",
                ),
                (
                    EmbedderConstructor(
                        ArnoldiEmbedder,
                        layers=["linear1", "linear2"],
                        projection_dim=None,
                        hessian_inverse_tol=0.0,
                    ),
                    "seq",
                ),
                (
                    EmbedderConstructor(
                        NaiveEmbedder,
                        layers=["linear1", "linear2"],
                        projection_dim=None,
                        hessian_inverse_tol=0.0,
                    ),
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
                    "conv",
                ),
                (
                    EmbedderConstructor(
                        FastKFACEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        # hessian_inverse_tol=0.0,
                        hessian_inverse_tol=-1e-2,
                        hessian_reg=1e-8,
                        projection_dim=50,
                    ),
                    "conv",
                ),
                (
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
                    "conv",
                ),
                (
                    EmbedderConstructor(
                        KFACEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        layer_projection_dim=100,
                        independent_factors=False,
                        # hessian_inverse_tol=0.0,
                        hessian_inverse_tol=-1e-2,
                        hessian_reg=1e-8,
                    ),
                    "conv",
                ),
                (
                    EmbedderConstructor(
                        ArnoldiEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        projection_dim=100,
                        # hessian_inverse_tol=0.0,
                        hessian_inverse_tol=-1e-2,
                        hessian_reg=1e-8,
                    ),
                    "conv",
                ),
                (
                    EmbedderConstructor(
                        NaiveEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        projection_dim=100,
                        # hessian_inverse_tol=0.0,
                        hessian_inverse_tol=-1e-2,
                        hessian_reg=1e-8,
                    ),
                    "conv",
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
                (
                    EmbedderConstructor(
                        PCAGradientEmbedder,
                        layers=["linear1", "conv"],
                        # layers=["linear1"],
                        projection_dim=5,
                        # hessian_inverse_tol=0.0,
                        # hessian_inverse_tol=-1e-2,
                        # hessian_reg=1e-8,
                    ),
                    "conv",
                ),
            ]
            for unpack_inputs in [
                False,
                True,
            ]
            for init_projection_on_cpu in (
                [True, False]
                if embedder_constructor.constructor_class
                in [FastKFACEmbedder, KFACEmbedder, ArnoldiEmbedder, NaiveEmbedder]
                else [None]
            )
            for load_projection_on_cpu in (
                [True, False]
                if embedder_constructor.constructor_class
                in [FastKFACEmbedder, KFACEmbedder, ArnoldiEmbedder, NaiveEmbedder]
                else [None]
            )
            for (reduction, sample_wise_grads_per_batch) in (
                [["sum", True], ["none", False]]
                if embedder_constructor.constructor_class not in [FastKFACEmbedder]
                else [["sum", None]]
            )
        ],
        name_func=build_test_name_func(),
    )
    def test_save_load_consistent(
        self,
        embedder_constructor: Callable,
        unpack_inputs,
        use_gpu: Union[bool, str],
        model_type: str,
        init_projection_on_cpu: Optional[bool],
        load_projection_on_cpu: Optional[bool],
        reduction: str,
        sample_wise_grads_per_batch: Optional[bool],
    ):
        """
        tests that directly computing embeddings and saving the results in `fit`,
        loading, then computing embeddings gives the same results.  also tests
        that calling `reset` and then `predict` without calling `load` results in a
        `NotFitException`.
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
        criterion = nn.MSELoss(reduction=reduction)
        
        embedder_constructor_kwargs = {
            "model": net,
            "loss_fn": criterion,
        }

        if init_projection_on_cpu is not None:
            embedder_constructor_kwargs["projection_on_cpu"] = init_projection_on_cpu
        if sample_wise_grads_per_batch is not None:
            embedder_constructor_kwargs["sample_wise_grads_per_batch"] = (
                sample_wise_grads_per_batch
            )

        embedder = embedder_constructor(**embedder_constructor_kwargs)

        embedder.fit(train_dataloader)

        test_dataloader = _format_inputs_dataset(
            (test_samples, test_labels)
            if not unpack_inputs
            else (*test_samples, test_labels)
        )

        embeddings_1 = embedder.predict(test_dataloader)

        with tempfile.NamedTemporaryFile() as tmp:
            embedder.save(tmp.name)
            embedder.reset()
            if not isinstance(embedder, GradientEmbedder):
                # `GradientEmbedder` does not need `fit` to be called
                self.assertRaises(NotFitException, embedder.predict, test_dataloader)
            if load_projection_on_cpu is not None:
                embedder.load(tmp.name, load_projection_on_cpu)
            else:
                embedder.load(tmp.name)
            embeddings_2 = embedder.predict(test_dataloader)

        assertTensorAlmostEqual(
            self, embeddings_1, embeddings_2, delta=1e-5, mode="sum"
        )
