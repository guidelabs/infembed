from unittest import TestCase
from infembed.clusterer._core.rule_clusterer import RuleClusterer
from infembed.clusterer._core.sklearn_clusterer import SklearnClusterer
from ...utils.common import build_test_name_func
from parameterized import parameterized
from infembed.clusterer._core.clusterer_base import ClustererBase
from .._utils.common import (
    assertClusteringEqual,
    get_random_embeddings_and_metadata,
)
from sklearn.cluster import KMeans


def _accuracy(data):
    return (data.metadata["prediction_label"] == data.metadata["label"]).mean()


def _size(data):
    return len(data)


class TestFitPredict(TestCase):
    @parameterized.expand(
        [
            RuleClusterer(
                clusterer_getter=lambda n_clusters: SklearnClusterer(
                    KMeans(n_clusters=n_clusters)
                ),
                cluster_rule=lambda data: _accuracy(data) < 0.3,
                stopping_rule=lambda data: _size(data) < 5,
                max_depth=5,
                branching_factor=2,
            ),
        ],
        name_func=build_test_name_func(),
    )
    def test_fit_predict_consistency(self, clusterer: ClustererBase):
        r"""
        tests that `fit_predict` gives same results as calling `fit` and `predict`
        separately.
        """
        data = get_random_embeddings_and_metadata()
        clustering_1 = clusterer.fit_predict(data)
        clustering_2 = clusterer.fit(data).predict(data)
        assert len(clustering_1) > 0 # boring to compare empty clustering
        assertClusteringEqual(self, clustering_1, clustering_2)
