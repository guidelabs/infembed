from unittest import TestCase
from infembed.embedder._core.kfac_embedder import KFACEmbedder
from .._utils.common import (
    get_random_model_and_data,
)
from parameterized import parameterized
from ...utils.common import assertTensorAlmostEqual, build_test_name_func
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


class TestKFACEmbedder(TestCase):
    @parameterized.expand(
        [
            True,
            False,
        ],
        name_func=build_test_name_func(),
    )
    def test_KFAC_least_squares_regression(self, independent_factors: bool):
        """
        tests that `KFACEmbedder`, without any dimension reduction (i.e.
        `layer_projection_dim` is None) is correct on linear regression.  "correct"
        means to apply the formula on page 17 of https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2022/readings/L04_second_order.pdf
        this formula is what `KFACEmbedder` uses, so there is some circular logic, but
        this does test that the formula was implemented correctly.
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

        embedder = KFACEmbedder(
            model=net,
            layers=["linear"],
            loss_fn=nn.MSELoss(reduction="sum"),
            sample_wise_grads_per_batch=True,
            layer_projection_dim=None,
            hessian_reg=1e-4,
            hessian_inverse_tol=1e-5,
            independent_factors=independent_factors,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
        embedder.fit(train_dataloader)

        # here we consider linear regression, so that loss is
        # $\sum_i (w'x_i - y_i)^2$.

        # first, directly calculate the influence between pairs of training examples
        x, y = next(iter(train_dataloader))
        h = net(x)

        # get "hessian" of matrix in 2D form
        output_gradients = y - h

        if not independent_factors:
            G = torch.mean(
                torch.stack(
                    [
                        torch.outer(_x, _x) * (output_gradient**2)
                        for (_x, output_gradient) in zip(x, output_gradients)
                    ],
                    dim=0,
                ),
                dim=0,
            )
        else:
            G = torch.mean(
                torch.stack([torch.outer(_x, _x) for _x in x], dim=0), dim=0
            ) * torch.mean(output_gradients**2)

        # get gradients of matrix in 1D form
        g = (x * output_gradients).T  # examples are columns

        # calculate influences
        manual_influences = g.T @ torch.linalg.pinv(G) @ g

        # second, calculate the same influence using the embeddings
        embeddings = embedder.predict(train_dataloader)
        embedder_influences = embeddings @ embeddings.T

        # the two influences should be equal
        assertTensorAlmostEqual(self, manual_influences, embedder_influences, 1e-2, "max")