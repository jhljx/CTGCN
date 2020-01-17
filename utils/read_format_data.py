import networkx as nx
import pandas as pd


def read_edgelist_from_dataframe(filename, full_node_list):
    dataframe = pd.read_csv(filename, sep='\t')
    # dataframe['weight'] = 1.0
    graph = nx.from_pandas_edgelist(dataframe, "from_id", "to_id", edge_attr='weight',
                                    create_using=nx.Graph)
    graph.add_nodes_from(full_node_list)
    return graph
