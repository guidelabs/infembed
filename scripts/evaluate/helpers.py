import numpy as np, pandas as pd
from collections import defaultdict


### define different metrics ###
def _blindspot_precision(p, t):
    return len(p.intersection(t)) / len(p)


def _blindspot_recall(precision_threshold, ps, t):
    covers = set()
    for p in ps:
        if _blindspot_precision(p, t) > precision_threshold:
            covers = covers.union(p.intersection(t))
    return len(covers) / len(t)


def _discovery_rate(covers_threshold, precision_threshold, ps, ts):
    return np.mean(
        [_blindspot_recall(precision_threshold, ps, t) > covers_threshold for t in ts]
    )


def _false_discovery_rate(precision_threshold, ps, ts):
    return np.mean(
        [
            max([_blindspot_precision(p, t) for t in ts]) < precision_threshold
            for p in ps
        ]
    )


def _get_ps_from_df(df):
    # 'cluster' column contains single cluster indexes
    return [set(_df.index) for (_, _df) in df.groupby("cluster")]


def _get_ts_from_df(df):
    # 'blindspot' column contains lists of indices
    l = defaultdict(list)
    for i, blindspots in df["blindspot"].items():
        for blindspot in blindspots:
            l[blindspot].append(i)
    return list(map(set, l.values()))
    # return [set(_df.index) for (_, _df) in df.groupby("blindspot")]


def discovery_rate(covers_threshold, precision_threshold, df):
    return _discovery_rate(
        covers_threshold, precision_threshold, _get_ps_from_df(df), _get_ts_from_df(df)
    )


def false_discovery_rate(precision_threshold, df):
    return _false_discovery_rate(
        precision_threshold, _get_ps_from_df(df), _get_ts_from_df(df)
    )


def is_in_blindspot_auc(df):
    """
    auc at prioritizing whether an example is in blindspot, if sort clusters by proportion
    that are in blindspot
    """
    _df = pd.DataFrame(
        {
            "is_in_blindspot": df["blindspot"].apply(lambda x: len(x) > 0),
            "cluster": df["cluster"],
        }
    )
    def add_proportion(__df):
        __df = __df.copy()
        __df['proportion_in_blindspot'] = __df['is_in_blindspot'].mean()
        return __df
    __df = _df.groupby('cluster').apply(add_proportion)
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(__df['is_in_blindspot'], __df['proportion_in_blindspot'])


def get_metrics(df, metric_infos):
    """
    given metric infos, applies them to return a dataframe.  metrics are a function of
    the dataframe
    """
    return pd.DataFrame(
        [
            {
                "metric_name": metric_info["metric_name"],
                "metric_value": metric_info["metric"](df=df),
            }
            for metric_info in metric_infos
        ]
    )