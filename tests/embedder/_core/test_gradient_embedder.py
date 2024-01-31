from typing import Callable, Union
from unittest import TestCase
from infembed.embedder._core.gradient_embedder import GradientEmbedder
from .._utils.common import get_random_model_and_data
from ...utils.common import assertTensorAlmostEqual
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


class TestGradientEmbedder(TestCase):
    def test_gradient_linear_regression(self):
        """
        tests that `GradientEmbedder returns the right gradients for linear regression
        """
        (
            net,
            train_dataset,
            _,
            _,
        ) = get_random_model_and_data(
            unpack_inputs=False,
            use_gpu=False,
            model_type="one_layer_linear",
        )
        embedder = GradientEmbedder(
            model=net,
            layers=["linear"],
            loss_fn=nn.MSELoss(reduction="sum"),
            sample_wise_grads_per_batch=True,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
        embedder.fit(train_dataloader)

        # here we consider linear regression, so that loss is
        # $\sum_i (w'x_i - y_i)^2$.

        # first, directly calculate the gradient
        x, y = next(iter(train_dataloader))
        h = net(x)

        manual_g = 2 * x * (y - h)

        # second, calculate the gradients using the embedder
        embedder_g = embedder.predict(train_dataloader)

        # compare the two gradients.  since elements in them may correspond to
        # different parameters, just check that pairwise dot-products of the two
        # gradients are equal
        manual_dot_products = manual_g @ manual_g.T
        embedder_dot_products = embedder_g @ embedder_g.T
        assertTensorAlmostEqual(
            self, manual_dot_products, embedder_dot_products, 1e-4, "max"
        )
