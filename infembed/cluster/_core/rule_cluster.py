from dataclasses import dataclass
from typing import Callable, List, Optional
from infembed.clusterer._core.clusterer_base import ClustererBase
import pandas as pd
from infembed.utils.common import Data
from infembed.clusterer._utils.common import get_canonical_clustering


@dataclass
class _Node:
    """
    This represents a node in the tree used to partition data.  It can optionally store
    the indices of the examples assigned to it when fitting, i.e. building the tree.
    """

    clusterer: ClustererBase
    children: Optional[List["Node"]]
    fit_indices: Optional[List[int]] = None


def _node_find(
    clusterer_getter,
    data,
    indices,
    depth,
    depth_to_K,
    cluster_rule,
    stopping_rule,
):
    """
    returns node, which is represented by cluster object, and list of nodes, which can have None elements
    can also return None, if nothing matches
    """
    if cluster_rule(data):
        return _Node(clusterer=None, children=None, fit_indices=indices)
    if depth >= len(depth_to_K) or stopping_rule(data):
        return None

    clusterer = clusterer_getter(depth_to_K[depth])
    # using `fit_predict` is safer, because not all implementations of `ClustererBase`
    # implement `fit` and `predict`, separately
    clusters = clusterer.fit_predict(data)
    cluster_datas = [data[cluster] for cluster in clusters]
    cluster_indicess = [[indices[i] for i in cluster] for cluster in clusters]

    children = [None for _ in range(len(clusters))]
    for (k, (cluster_data, cluster_indices)) in enumerate(zip(cluster_datas, cluster_indicess)):
        children[k] = _node_find(
            clusterer_getter,
            cluster_data,
            cluster_indices,
            depth + 1,
            depth_to_K,
            cluster_rule,
            stopping_rule,
        )

    if sum([child is not None for child in children]) > 0:
        return _Node(clusterer, children)
    else:
        return None


def _get_trail_to_predict_cluster(data, indices, node, trail):
    """
    returns dictionary mapping from the "trail" to a cluster, i.e. the index of the
    branch to go down at each node on the path to the node representing the cluster,
    to the indices of the examples in the clustering of `data`
    """
    if node is None:
        return {}
    if node.clusterer is None:
        return {trail: indices}
    clusters = node.clusterer.predict(data)
    cluster_datas = [data[cluster] for cluster in clusters]
    cluster_indicess = [[indices[i] for i in cluster] for cluster in clusters]
    d = {}
    for k, (child, cluster_data, cluster_indices) in enumerate(
        zip(node.children, cluster_datas, cluster_indicess)
    ):
        d = {
            **d,
            **_get_trail_to_predict_cluster(
                cluster_data,
                cluster_indices,
                child,
                tuple(list(trail) + [k]),
            ),
        }
    return d


def _get_trail_to_fit_cluster(node, trail):
    """
    returns dictionary mapping from the "trail" to a cluster, i.e. the index of the
    branch to go down at each node on the path to the node representing the cluster,
    to the indices of the examples in the clustering of fitting data, which was
    constructed at fitting time.
    """
    if node is None:
        return {}
    if node.clusterer is None:
        return {trail: node.fit_indices}
    d = {}
    for (k, child) in enumerate(node.children):
        d = {**d,  **_get_trail_to_fit_cluster(child,tuple(list(trail) + [k]))}
    return d


class RuleClusterer(ClustererBase):
    r"""
    Implementation of `ClustererBase` that recursively clusters examples using
    another `ClustererBase` implementation until the clusters satisfy a provided rule.
    The rule is assumed to be a function of provided metadata.  This method corresponds
    to the 'ImfEmbed-Rule` method in https://arxiv.org/abs/2312.04712.

    TODO: handle base clusters that only implement `fit_predict`, like `DBScan`.
    """

    def __init__(
        self,
        clusterer_getter: Callable[[int], ClustererBase],
        cluster_rule: Callable,
        stopping_rule: Callable,
        max_depth: int,
        branching_factor: int,
    ):
        self.clusterer_getter, self.cluster_rule, self.stopping_rule = (
            clusterer_getter,
            cluster_rule,
            stopping_rule,
        )
        self.depth_to_K = [branching_factor for _ in range(max_depth)]

    def fit(self, data: Data):
        r"""
        Does the prepratory computation needed to assign new examples to clusters.  In
        particular, it creates a tree.  If a node is not a leaf, it uses a clusterer
        to route the example it receives.  If a node is a leaf, the examples that are
        routed to it represent a cluster satisfying the provided rule.

        Args:
            data (Data): `Data` representing the examples used for doing the
                    prepratory computation.
        """
        self.root = _node_find(
            self.clusterer_getter,
            data,
            list(range(len(data))),
            0,
            self.depth_to_K,
            self.cluster_rule,
            self.stopping_rule,
        )
        return self

    def predict(self, data: Data):
        r"""
        Assigns the examples in `embeddings` to clusters.

        Args:
            data (Data): `Data` representing the examples to assign to clusters.
        """
        d = _get_trail_to_predict_cluster(data, list(range(len(data))), self.root, tuple())
        return get_canonical_clustering(list(d.values()))
    
    def fit_predict(self, data: Data):
        r"""
        Assigns the examples represented by `embeddings` to clusters, after doing the
        prepratory computation.

        Args:
            data (Data): `Data` representing the examples to assign to clusters.
        """
        self.fit(data)
        d = _get_trail_to_fit_cluster(self.root, tuple())
        return get_canonical_clustering(list(d.values()))