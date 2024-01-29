from abc import ABC, abstractmethod
from typing import List
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


class ClusterAUCDisplayer(Displayer):
    """
    see how well clustering prioritizes a given column in metadata
    """
    def __init__(self, col: str):
        self.col = col

    def __call__(self, clusters: List[List[int]], data: Data):
        # make df with cluster and `col` as columns
        d = {}
        for k, cluster in enumerate(clusters):
            for index in cluster:
                d[index] = k
        # print(pd.Series(d))
        # print(data.metadata[self.col])
        #import pdb
        #pdb.set_trace()
        df = pd.DataFrame({'cluster': pd.Series(d), self.col: data.metadata[self.col]})

        def add_proportion(_df):
            _df = _df.copy()
            _df['proportion'] = _df[self.col].mean()
            return _df
        df = df.groupby('cluster').apply(add_proportion)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(df[self.col], df['proportion'])
        print(f"AUC for {self.col}: {auc}")
        


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
        data: Data,
    ):
        assert isinstance(data.metadata, pd.DataFrame)
        corrects = data.metadata.iloc[cluster][self.prediction_col] == data.metadata.iloc[cluster][self.label_col]
        print(f"accuracy: {corrects.mean():.2f} ({corrects.sum()}/{len(corrects)})")


class DisplayPredictionAndLabels(SingleClusterDisplayer):
    def __init__(self, prediction_col='prediction_label', label_col='label'):
        self.prediction_col, self.label_col = prediction_col, label_col

    def __call__(
        self,
        cluster: List[int],
        data: Data,
    ):
        assert data.metadata is not None
        prediction_counts = data.metadata.iloc[cluster][self.prediction_col].value_counts()
        label_counts = data.metadata.iloc[cluster][self.label_col].value_counts()
        print(f"prediction: {dict(prediction_counts)}")
        print(f"label: {dict(label_counts)}")


class DisplayCounts(SingleClusterDisplayer):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def __call__(
        self,
        cluster: List[int],
        data: Data,
    ):
        assert data.metadata is not None
        for col in self.cols:
            counts = data.metadata.iloc[cluster][col].value_counts().sort_index()
            print(f"{col}: {dict(counts)}")