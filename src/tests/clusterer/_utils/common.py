from infembed.utils.common import Data
import torch
import pandas as pd


def assertClusteringEqual(test, actual, expected):
    assert len(actual) == len(expected), f"clusterings are of different lengths"
    return actual == expected  # take advantage that both are lists of lists of int


def get_random_embeddings_and_metadata() -> Data:
    size = 1000
    torch.manual_seed(42)
    return Data(
        embeddings=torch.randn((size, 5)),
        metadata=pd.DataFrame(
            {
                "prediction_label": torch.rand(size) < 0.5,
                "label": torch.rand(size) < 0.5,
            }
        ),
    )
