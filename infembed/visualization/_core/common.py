from abc import ABC, abstractmethod
from typing import Callable, List, Optional
import pandas as pd
from infembed.utils.common import Data
import matplotlib.pyplot as plt
import torch


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
            header = f"""
###############
   cluster {k} 
###############
            """
            print(header)
            for displayer in self.single_cluster_displayers:
                displayer(cluster, data)
                print()


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
    def __init__(self, cols: List[str], ignore_threshold: Optional[int] = None):
        self.cols = cols
        self.ignore_threshold = ignore_threshold

    def __call__(
        self,
        cluster: List[int],
        data: Data,
    ):
        assert data.metadata is not None
        for col in self.cols:
            counts = data.metadata.iloc[cluster][col].value_counts().sort_values(ascending=False)
            if self.ignore_threshold is not None:
                counts = counts[counts > self.ignore_threshold]
            print(f"{col} counts: {dict(counts)}")


class SingleExampleDisplayer(ABC):
    @abstractmethod
    def __call__(self, i: int, data: Data):
        pass


class DisplayMetadata(SingleExampleDisplayer):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def __call__(self, i: int, data: Data):
        print(pd.DataFrame({i: data.metadata[self.cols].iloc[i]}).T)


class DisplayPIL(SingleExampleDisplayer):
    def __init__(self, height=200):
        self.height = height

    def __call__(self, i: int, data: Data):
        #fig, ax = plt.subplots()
        # fig = plt.figure(figsize=(3,4))
        # ax = fig.add_subplot(1,1,1)
        # assume dataset has the image in 0-th position
        image = data.dataset[i][0]
        width, height = image.size
        new_height = self.height
        new_width = int(width * new_height / height)
        image = image.resize((new_width, new_height))
        image.show()
        # import pdb
        # pdb.set_trace()
        # data.dataset[i][0].show()
        # from torchvision.transforms.functional import pil_to_tensor
        # .permute(1, 2, 0)
        # ax.imshow(pil_to_tensor(data.dataset[i][0]).permute(1, 2, 0))
        # ax.imshow(torch.clip(pil_to_tensor(data.dataset[i][0]).permute(1, 2, 0), 0, 1))
        # fig.show()
        #plt.close(fig)


class DisplaySingleExamples(SingleClusterDisplayer):
    def __init__(self, single_example_displayers: List[SingleExampleDisplayer], limit=None, condition: Optional[Callable] = None):
        self.single_example_displayers = single_example_displayers
        self.limit, self.condition = limit, condition

    def __call__(self, cluster: List[int], data: Data):
        num = 0
        for i in cluster:
            if self.limit is not None and num >= self.limit:
                break
            if self.condition(i, data):
                header = f"""
### example {i} ###
"""
                print(header)
                for displayer in self.single_example_displayers:
                    displayer(i, data)
                num += 1
                print()