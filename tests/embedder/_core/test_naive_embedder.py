from typing import Callable, Union
from unittest import TestCase
from infembed.embedder._core.fast_kfac_embedder import FastKFACEmbedder
from infembed.embedder._core.kfac_embedder import KFACEmbedder
from infembed.embedder._core.naive_embedder import NaiveEmbedder
from infembed.embedder._utils.common import _format_inputs_dataset
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


class TestNaiveEmbedder(TestCase):
    @parameterized.expand(
        [
            (reduction, unpack_inputs, use_gpu, sample_wise_grads_per_batch)
            for (reduction, sample_wise_grads_per_batch) in [
                ('sum', True),
                ('mean', True),
                ('none', False),
            ]
            for unpack_inputs in [
                True,
                False,
            ]
            for use_gpu in USE_GPU_LIST
        ],
        name_func=build_test_name_func(),
    )
    def test_matches_linear_regression(
        self,
        reduction: str,
        unpack_inputs: bool,
        use_gpu: Union[bool, str],
        sample_wise_grads_per_batch: bool,
    ):
        (
            net,
            train_dataset,
            test_samples,
            test_labels,
        ) = get_random_model_and_data(
            unpack_inputs=unpack_inputs,
            use_gpu=use_gpu,
            model_type="trained_linear",
        )

        embedder = NaiveEmbedder(
            model=net,
            layers=['linear'],
            projection_dim=False,
            loss_fn=nn.MSELoss(reduction=reduction),
            sample_wise_grads_per_batch=sample_wise_grads_per_batch,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
        embedder.fit(train_dataloader)

        if not unpack_inputs:
            tensor_train_samples, train_labels = next(iter(train_dataloader))
        else:
            # import pdb
            # pdb.set_trace()
            tensor_train_samples_1, tensor_train_samples_2, train_labels = next(iter(train_dataloader))
            tensor_train_samples = torch.cat([tensor_train_samples_1, tensor_train_samples_2], dim=1)

        # hessian at optimal parameters is 2 * X'X, where X is the feature matrix
        # of the examples used for calculating the hessian.
        # this is based on https://math.stackexchange.com/questions/2864585/hessian-on-linear-least-squares-problem # noqa: E501
        # and multiplying by 2, since we optimize squared error,
        # not 1/2 squared error.
        hessian = torch.matmul(tensor_train_samples.T, tensor_train_samples) * 2
        hessian = hessian + (
            torch.eye(len(hessian)).to(device=hessian.device) * 1e-4
        )
        # version = _parse_version(torch.__version__)
        if False and version < (1, 8):
            hessian_inverse = torch.pinverse(hessian, rcond=1e-4)
        else:
            hessian_inverse = torch.linalg.pinv(hessian, rcond=1e-4)

        # gradient for an example is 2 * features * error

        # compute train gradients
        train_predictions = torch.cat(
            [net(*batch[:-1]) for batch in train_dataloader], dim=0
        )
        train_gradients = (
            (train_predictions - train_labels) * tensor_train_samples * 2
        )

        # compute test gradients
        tensor_test_samples = (
            test_samples if not unpack_inputs else torch.cat(test_samples, dim=1)
        )
        test_predictions = (
            net(test_samples) if not unpack_inputs else net(*test_samples)
        )
        test_gradients = (test_predictions - test_labels) * tensor_test_samples * 2

        # compute pairwise influences, analytically
        analytical_train_test_influences = torch.matmul(
            torch.matmul(test_gradients, hessian_inverse), train_gradients.T
        )

        # compute pairwise influences using embeddings
        train_embeddings = embedder.predict(train_dataloader)
        test_batch = (test_samples, test_labels) if not unpack_inputs else (*test_samples, test_labels)
        test_embeddings = embedder.predict(_format_inputs_dataset(test_batch))
        embedding_train_test_influences = torch.matmul(test_embeddings, train_embeddings.T)

        assertTensorAlmostEqual(
            self,
            embedding_train_test_influences,
            analytical_train_test_influences,
            delta=1e-3,
            mode="max",
        )