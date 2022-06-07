import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from logging import info
from sklearn import preprocessing
from nilearn.plotting import view_markers


def _check_array(graphs, labels, idx) -> list:
    """Check ensures input is an array.

    Args:
        graphs (_type_): iterrable type of graph information
        labels (_type_): iterrable type of labels
        idx (_type_): index of graphs

    Returns:
        list: array of graphs, labels, and indices.
    """
    graphs = np.asarray(graphs)
    labels = np.asarray(labels)
    idx = np.asarray(idx)

    if (graphs.shape != labels.shape or
       graphs.shape != idx.shape):
        raise ValueError('Inputs must be arrays of the same shape.')
    return graphs, labels, idx


def _feasable_graph(graph: nx.Graph,
                    weakly_connected: bool = False) -> nx.Graph:
    """Ensures graph structure is feasable.

    Args:
        graph (nx.Graph): networkx graph.
        weakly_connected (bool, optional): ensure graph is weakly connected.
            Defaults to False.

    Returns:
        nx.Graph: feasable graph.
    """
    degree = sum(n[1] for n in list(nx.degree(graph)))

    if degree == 0:
        return None
    if weakly_connected:
        graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def _feasable_dict(graph: dict, **kwargs) -> dict:
    """Ensures graph metric is feasable.

    Args:
        graph (dict): graph metric.

    Returns:
        dict: feasable graphs.
    """
    n_values = len(graph)

    if n_values == 0:
        return None
    else:
        return graph


def _check_consitency(graphs: list, weakly_connected: bool = False) -> list:
    """Check if graphs are consistent with embedding assumptions.

    Args:
        graphs (list): list of networkx graphs.
        weakly_connected (bool, optional): ensure graph is weakly connected.
            Defaults to False.

    Returns:
        list: consisten graphs and list of graphs to remove.
    """    ""

    if (all(isinstance(d, nx.Graph) for d in graphs)):
        check = _feasable_graph
    elif (all(isinstance(d, dict) for d in graphs)):
        check = _feasable_dict
    else:
        raise ValueError('Graphs must be either networkx graphs or dicts.')

    pop = []
    for idx, graph in enumerate(graphs):
        corr_graph = check(graph, weakly_connected=weakly_connected)
        if corr_graph is not None:
            graphs[idx] = corr_graph
        else:
            pop.append(idx)

    return graphs, pop


def _remove_graphs(pops: list, graphs: list, labels: list, idx: list) -> list:
    """Remove graphs from list.

    Args:
        pops (list): list of indices to remove.
        graphs (list): list of networkx graphs.
        labels (list): list of labels.
        idx (list): list of indices.

    Returns:
        list: remaining graphs, labels, and indices.
    """
    idx = [i for idx, i in enumerate(idx) if idx not in pops]
    labels = [l for idx, l in enumerate(labels) if idx not in pops]
    graphs = [g for idx, g in enumerate(graphs) if idx not in pops]
    graphs, labels, idx = _check_array(graphs, labels, idx)
    return graphs, labels, idx


class NetworkEmbedding(object):
    """Creates Embedding from graph data"""

    def __init__(self,
                 embedding):
        """Initializes embedding"""
        self.embedding = embedding
        self._excluded_graphs = None

    def _set_obj_params(self, **parameters):
        """Sets object parameters. """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _get_preprocessing(self):
        """Gets preprocessing method"""

        type_preprocessing = {'Embedding': self._embedding_preprocessing}
        function_preprocessing = type_preprocessing.get(
            self.embedding.__class__.__name__, 'otherwise'
        )
        return function_preprocessing

    def _embedding_preprocessing(self):
        """Preprocessing for embedding class"""

        if not (all(isinstance(d, dict) for d in self.data)):
            graphs = self.data
            raise ValueError('Data must be a list of dicts')

        graphs = self.data
        graphs, pops = _check_consitency(graphs, False)

        if len(pops) > 0 and self.remove_empty:
            self._excluded_graphs = pops
            graphs, self.labels, self.labels = (
                _remove_graphs(pops, graphs, self.labels, self.idx)
            )
            info(f'{len(pops)} graphs were removed from the dataset')

        self._data = graphs

    @staticmethod
    def encoder(input_labels: list, input_confounds: list = None) -> tuple:
        """Encodes labels.

        Args:
            input_labels (list): iterable labels.
            input_confounds (list, optional): iterable confounds.

        Returns:
            tuple: encoded labels, confounds, and classes.
        """

        # labels
        le_labels = preprocessing.LabelEncoder()
        le_labels.fit(input_labels)
        classes = list(le_labels.classes_)
        labels = le_labels.transform(input_labels)

        # confounds
        encoded_confounds = None
        if input_confounds is not None:
            le_confounds = preprocessing.LabelEncoder()
            le_confounds.fit(input_confounds)
            encoded_confounds = le_confounds.transform(input_confounds)

            # reshape confounds
            encoded_confounds = encoded_confounds.reshape(-1, 1)

        return labels, encoded_confounds, classes

    def plot_lookup(self, feature: int, threshold: float, mni: pd.DataFrame):
        """Plots lookup / factorization of embedding.

        Args:
            feature (int): id of feature to plot.
            threshold (float): threshold for features with
                relevant contributions.
            mni (pd.DataFrame): MNI coordinates.

        Returns:
            list: glas brain plots.
        """

        if (not hasattr(self, 'embedding') or
           not hasattr(self.embedding, '_nodes')):
            pass

        relevant_mni = mni.loc[self.embedding._nodes, :]
        fact = self.embedding.get_factorization()
        from sklearn.preprocessing import MinMaxScaler

        scaled_fact = (
            MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(fact)
        )
        thr = np.quantile(np.concatenate(scaled_fact).ravel(), threshold)
        scaled_fact[scaled_fact < thr] = 0
        relevant_mni['contributes'] = scaled_fact[feature] > 0

        cmap = matplotlib.cm.get_cmap('GnBu')
        relevant_mni['colors'] = [cmap(v) for v in scaled_fact[feature]]
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        fig, axs = plt.subplots(nrows=1, figsize=(6.4, 0.35))
        axs.imshow(gradient, aspect='auto', cmap=cmap)
        axs.set_axis_off()

        set1 = relevant_mni.loc[(relevant_mni.set == 1) &
                                (relevant_mni.contributes),
                                ['X', 'Y', 'Z']]
        set2 = relevant_mni.loc[(relevant_mni.set == 2) &
                                (relevant_mni.contributes),
                                ['X', 'Y', 'Z']]

        colors_set1 = relevant_mni.loc[(relevant_mni.set == 1) &
                                       (relevant_mni.contributes),
                                       'colors']
        colors_set2 = relevant_mni.loc[(relevant_mni.set == 2) &
                                       (relevant_mni.contributes),
                                       'colors']

        set1_plt = view_markers(set1,
                                marker_color=colors_set1,
                                marker_size=10,
                                marker_labels=list(set1.index.values))
        set2_plt = view_markers(set2,
                                marker_color=colors_set2,
                                marker_size=10,
                                marker_labels=list(set2.index.values))
        return set1_plt, set2_plt

    def fit_transform(self,
                      data: list,
                      labels: list,
                      idx: list,
                      remove_empty: bool = False,
                      **kwargs) -> np.ndarray:
        """Fits and transforms data.

        Args:
            data (list): iterable of graphs.
            labels (list): iterable of labels.
            idx (list): iterable index.
            remove_empty (bool, optional): remove empty graphs.
                Defaults to False.

        Returns:
            np.ndarray: embedding.
        """

        data, labels, idx = _check_array(data, labels, idx)

        self.data = data
        self.labels = labels
        self.idx = idx
        self.remove_empty = remove_empty
        self._set_obj_params(**kwargs)

        processing = self._get_preprocessing()
        processing()

        self.embedding.fit(self._data)

        self.values = self.embedding.get_embedding()
        self.label_values, _, self.label_classes = self.encoder(self.labels)
        return self
