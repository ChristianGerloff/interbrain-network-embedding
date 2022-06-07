"""
Constructs interbrain netowks and provides
topolicy metrics.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from logging import warning
from dask import delayed

SEPERATOR = '_'
COORD_COLS = [['Channel_1_X', 'Channel_1_Z'],
              ['Channel_2_X', 'Channel_2_Z']]


def _check_channels(data: pd.DataFrame,
                    channel_cols: list,
                    coordinate_cols: list) -> list:
    """Check if all channels have available coordinates and channels.

    Args:
        data (pd.DataFrame): Dataframe with channels and coordinates.
        channel_cols (list):  List of channel names.
        coordinate_cols (list): List of coordinate names.

    Returns:
        list: Channel and coordinate column names and corresponding data.
    """
    cols = []

    for i in channel_cols:
        if i not in data.columns:
            raise ValueError(f'{i} not in network data available.')
        cols.append(i)

    if len(channel_cols) != len(coordinate_cols):
        raise ValueError('Number of channels and coordinates must be equal.')

    for i in coordinate_cols:
        if not set(i).issubset(data.columns):
            raise ValueError(f'{i} not in network data available.')
        cols += i

    data = data[cols].drop_duplicates()
    coordinates = dict()
    for ch_idx, channel in enumerate(channel_cols):
        coordinates.update(
            {
                row[channel]: row[coordinate_cols[ch_idx]].values
                for idx, row in data.iterrows()
            }
        )

    return data, coordinates


def _standardize(numerator: list,
                 denominator: list) -> list:
    """Standarize by possible channels
       to account for bad channels.

    Args:
        numerator (list): _description_
        denominator (list): _description_

    Returns:
        dictionary: degree values
    """
    n_idx, n_values = zip(*numerator)
    n_series = pd.Series(n_values, n_idx)
    d_idx, d_values = zip(*denominator)
    d_series = pd.Series(d_values, d_idx)

    div = n_series/d_series
    div = div.fillna(0)  # delete nans from deleted edges
    return div.to_dict()


def _calculate_metrics(list_graphs: list, max_degree: list) -> dict:
    """Calculate topological metrics for a list of graphs.

    Args:
        list_graphs (list): list of graphs
        max_degree (list): list of max nodal degrees per graph

    Returns:
        dict: _description_
    """
    metrics = {
        'metric': [_standardize(list(nx.degree(g)), max_degree)
                   for g in list_graphs],
        'n_possible_edges': [max_degree for g in list_graphs],
    }
    return metrics


def _complete_table(table: pd.DataFrame,
                    start_names: list = [],
                    start_values: list = [],
                    end_names: list = [],
                    end_values: list = []) -> pd.DataFrame:
    """Complete a table with additional metadata.

    Args:
        table (pd.DataFrame): input table
        start_names (list, optional): names of columns to be add
            at the start. Defaults to [].
        start_values (list, optional): values of start
            meta data. Defaults to [].
        end_names (list, optional): names of columns
            to be add at the end. Defaults to [].
        end_values (list, optional): values of metadata
            at the end. Defaults to [].

    Returns:
        pd.DataFrame: completed dataframe
    """

    if any(start_names):
        for i, item in enumerate(zip(start_names, start_values)):
            name, value = item
            table.insert(i, name, value)
    if any(end_names):
        table = table.assign(**dict(zip(end_names, end_values)))
    return table


def construct_bipartite_graphs(data: pd.DataFrame,
                               graph_id: int = None,
                               estimator: str = 'WCO',
                               channel_cols: list = ['Channel_1', 'Channel_2'],
                               path: Path = None,
                               coord_cols: list = COORD_COLS) -> pd.DataFrame:
    # Initialize result df
    result = pd.DataFrame(columns=['graph_type', 'weights'])

    # ensure that graph_id is not None
    if ((estimator + '_null_dist') not in data.columns and
       True not in data[estimator + '_null_dist'].values):
        result = _complete_table(
            result,
            ['graph_id', 'ID', 'conditions', 'estimator'],
            [graph_id, data['ID'].iloc[0], data['conditions'].iloc[0],
             estimator],
            ['probe_1', 'baseline', 'condition', 'partner'],
            [data['Probe_1'].iloc[0], data['Baseline'].iloc[0],
             data['Condition'].iloc[0], data['Partner'].iloc[0]]
        )
        return result

    # check if coordinates are available
    nodes, coordinates = _check_channels(data, channel_cols, coord_cols)

    # add all nodes (without bad channels)
    complete_g = nx.Graph()
    complete_g.add_nodes_from(nodes[channel_cols[0]],
                              bipartite=0)
    complete_g.add_nodes_from(nodes[channel_cols[1]],
                              bipartite=1)

    # set weights
    complete_g.add_weighted_edges_from(
            [
                (row[channel_cols[0]],
                 row[channel_cols[1]],
                 row[estimator])
                for idx, row in data.iterrows()
            ],
            weight='weight'
    )

    if data[estimator].isnull().values.any():
        warning(f'Some edges have no weight: {graph_id}')

    # degree of graph
    max_degree = list(nx.degree(complete_g))

    # reduce graph via null distribution
    remove_null_model = data[data[estimator + '_null_dist'] == False]
    remove_null_model = remove_null_model[[channel_cols[0], channel_cols[1]]]
    remove_null_model = remove_null_model.drop_duplicates()

    # reduced interbrain network
    ibn = complete_g.copy()
    ibn.remove_edges_from(
        [
            (row[channel_cols[0]], row[channel_cols[1]])
            for idx, row in remove_null_model.iterrows()
        ]
    )

    attr = nx.get_edge_attributes(ibn, 'weight')
    query = pd.DataFrame(
        [['interbrainnetwork', attr, coordinates]],
        columns=['graph_type', 'weights', 'pos']
    )
    result = pd.concat([result, query], ignore_index=True)
    list_graphs = [ibn]  # list of graphs

    # add topological metrics to result
    metric = _calculate_metrics(list_graphs, max_degree)
    result = result.assign(**metric)

    result = _complete_table(
        result,
        ['graph_id', 'ID', 'conditions', 'estimator'],
        [graph_id, data['ID'].iloc[0], data['conditions'].iloc[0],
         estimator],
        ['probe_1', 'baseline', 'condition', 'partner'],
        [data['Probe_1'].iloc[0], data['Baseline'].iloc[0],
         data['Condition'].iloc[0], data['Partner'].iloc[0]]
    )

    # save graphs
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        prefix_filename = (
            str(graph_id) + SEPERATOR +
            str(data['ID'].iloc[0]) + SEPERATOR +
            data['conditions'].iloc[0] + SEPERATOR +
            estimator + SEPERATOR +
            data['chromophore'].iloc[0]
        )
        for i, g in enumerate(list_graphs):
            filename = str(
                prefix_filename + SEPERATOR +
                result.loc[i, 'graph_type'] +
                '.gexf'
            )
            nx.write_gexf(g, path / filename)
            result.loc[i, 'filename'] = filename
    return result


@delayed
def calculate_topologies(data: pd.DataFrame,
                         estimator: str,
                         path: Path = None):

    graph = (
        data.groupby('signal_pair').apply(
            lambda r: construct_bipartite_graphs(
                r,
                graph_id=r.name,
                estimator=estimator,
                path=path
            )
        )
    )
    graph = graph.fillna(value=0)
    return graph
