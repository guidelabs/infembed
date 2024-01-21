from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from infembed.utils.common import Data


class Displayer(ABC):
    """
    Displays an entire set of clusters
    """

    @abstractmethod
    def __call__(
        self,
        clusters: List[List[int]],
        data: Data,
    ):
        pass


class SingleClusterDisplayer(ABC):
    """
    Displays a single cluster
    """

    @abstractmethod
    def __call__(
        self,
        cluster: List[int],
        data: Data,
    ):
        pass


class PerClusterDisplayer(Displayer):
    def __init__(self, single_cluster_displayers: List[SingleClusterDisplayer]):
        self.single_cluster_displayers = single_cluster_displayers

    def __call__(
        self,
        clusters: List[List[int]],
        data: Data,
    ):
        for k, cluster in enumerate(clusters):
            print(f"cluster #{k}")
            for displayer in self.single_cluster_displayers:
                displayer(cluster, data)


class DisplayAccuracy(SingleClusterDisplayer):
    def __init__(self, prediction_col='prediction_label', label_col='label'):
        self.prediction_col, self.label_col = prediction_col, label_col

    def __call__(
        self,
        cluster: List[int],
        data,
    ):
        assert isinstance(data.metadata, pd.DataFrame)
        corrects = data.metadata.iloc[cluster][self.prediction_col] == data.metadata.iloc[cluster][self.label_col]
        print(f"accuracy: {corrects.mean():.2f} ({corrects.sum()}/{len(corrects)})")


class DisplayPredictionAndLabels(SingleClusterDisplayer):
    def __init__(self, figsize=None, threshold=-1, num=10, prediction_col='prediction_label', label_col='label'):
        self.prediction_col, self.label_col = prediction_col, label_col

    def __call__(
        self,
        clusters: List[List[int]],
        data: Data,
    ):
        assert metadata is not None
        prediction_counts = metadata[self.prediction_col].value_counts()
        label_counts = metadata[self.label_col].value_counts()