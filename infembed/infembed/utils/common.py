from dataclasses import dataclass
import torch
from typing import Optional, List
import pandas as pd
from torch.utils.data import Dataset, Subset


@dataclass
class Data:
    """
    container for different data potentially useful for embedding, clustering, visualization
    """

    embeddings: Optional[torch.Tensor] = None
    metadata: Optional[pd.DataFrame] = None
    dataset: Optional[Dataset] = None

    def __getitem__(self, indices: List[int]):
        return Data(
            self.embeddings[indices],
            self.metadata.iloc[indices],
            Subset(self.dataset, indices),
        )

    def __len__(self):
        if self.embeddings is not None:
            return len(self.embeddings)
        elif self.metadata is not None:
            return len(self.metadata)
        elif self.dataset is not None:
            return len(self.dataset)


import cProfile
import io
import pstats
from pstats import SortKey


def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper
