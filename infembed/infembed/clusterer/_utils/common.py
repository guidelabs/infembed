from typing import List
import numpy as np
from collections import defaultdict


def get_canonical_clustering(clustering: List[List[int]]):
    clustering = list(map(sorted, clustering))
    clustering = sorted(clustering)
    return clustering


def _cluster_assignments_to_indices(assignments: np.ndarray) -> List[List[int]]:
    d = defaultdict(list)
    for (i, k) in enumerate(assignments):
        d[int(k)].append(i)
    return [d[k] for k in sorted(d.keys())]