from typing import List


def get_canonical_clustering(clustering: List[List[int]]):
    clustering = list(map(sorted, clustering))
    clustering = sorted(clustering)
    return clustering