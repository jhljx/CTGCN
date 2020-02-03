import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, multiprocessing, random
from RWTGCN.utils import check_and_make_path, wl_transform, sigmoid


class StructuralNetworkGenerator:
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    hop: int
    max_neighbor_num: int

    def __init__(self, base_path, input_folder, output_folder, node_file, hop=5, max_neighbor_num=100):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.hop = hop
        self.max_neighbor_num = max_neighbor_num

        check_and_make_path(self.output_base_path)
        tem_dir = ['structural_network']
        for tem in tem_dir:
            check_and_make_path(os.path.join(self.output_base_path, tem))

    def get_structural_network_all_time(self, worker=-1):
        print("getting all timestamps structural network adjacent...")

        f_list = os.listdir(self.input_base_path)
        length = len(f_list)

        if worker <= 0:
            for i, f_name in enumerate(f_list):
                self.get_structural_network(
                    input_file=os.path.join(self.input_base_path, f_name),
                    output_file=os.path.join(self.output_base_path, "structural_network", f_name),
                    file_num=length, i=i)

        else:
            worker = min(worker, length, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for i, f_name in enumerate(f_list):
                pool.apply_async(self.get_structural_network, (
                    os.path.join(self.input_base_path, f_name),
                    os.path.join(self.output_base_path, "structural_network", f_name), length, i,))

            pool.close()
            pool.join()
        print("got it...")

    def get_structural_network(self, input_file, output_file, file_num, i):
        print('\t', str(file_num - i), ' file(s) left')
        if os.path.exists(output_file):
            print('\t', output_file, "exist")
            print('\t', str(file_num - i), ' finished')
            return

        df_origin = pd.read_csv(input_file, sep="\t")
        node_num = len(self.full_node_list)
        nid2idx_dict = dict(zip(self.full_node_list, np.arange(node_num).tolist()))
        idx2nid_dict = dict(zip(np.arange(node_num).tolist(), self.full_node_list))
        df_origin['from_id'] = df_origin['from_id'].map(nid2idx_dict)
        df_origin['to_id'] = df_origin['to_id'].map(nid2idx_dict)
        spadj = sp.csr_matrix((df_origin['weight'], (df_origin['from_id'], df_origin['to_id'])),
                              shape=(node_num, node_num))
        del df_origin
        labels = np.ones(node_num)
        new_labels, cluster_dict = wl_transform(spadj, labels, cluster=True)
        df_colors = pd.DataFrame(new_labels, columns=[0])
        for i in range(1, self.hop):
            new_labels = wl_transform(spadj, new_labels, cluster=False)
            df_cur = pd.DataFrame(new_labels, columns=[i])
            df_colors = pd.concat([df_colors, df_cur], axis=1)
        df_colors['node'] = np.arange(node_num)

        structural_edges_dict = dict()
        def calc_structural_similarity(series):
            from_idx = nid2idx_dict[series['from_id']]
            to_idx = nid2idx_dict[series['to_id']]
            if (series['from_id'], series['to_id']) in structural_edges_dict:
                return
            from_arr = df_colors.loc[from_idx].values
            to_arr = df_colors.loc[to_idx].values
            weight = sigmoid(from_arr.dot(to_arr))
            key = (series['from_id'], series['to_id'])
            structural_edges_dict[key] = weight

        def get_structural_neigbors(series):
            node_idx = series['node']
            cluster_type = series[0]
            cluster_list = cluster_dict[cluster_type].copy()
            cluster_list.remove(node_idx)
            cluster_num = len(cluster_list)
            if cluster_num == 0:
                return
            cnt = random.randint(1, min(cluster_num, self.max_neighbor_num))
            sampled_nodes = random.sample(cluster_list, cnt)
            def map_func(val):
                return idx2nid_dict[val]
            sampled_nodes = list(map(map_func, sampled_nodes))
            df_sim = pd.DataFrame(sampled_nodes, columns=['to_id'])
            df_sim['from_id'] = idx2nid_dict[node_idx]
            df_sim.apply(calc_structural_similarity, axis=1)
            return
        df_colors.apply(get_structural_neigbors, axis=1)
        df_colors = df_colors.drop(['node'], axis=1)

        edge_arr = np.array(list(structural_edges_dict.keys()))
        weight_arr = np.array(list(structural_edges_dict.values())).reshape(-1, 1)
        data_arr = np.hstack((edge_arr, weight_arr))
        df_structural_edges = pd.DataFrame(data_arr, columns=['from_id', 'to_id', 'weight'])
        print('edge num: ', df_structural_edges.shape[0])
        df_structural_edges.to_csv(output_file, sep='\t', index=False, header=True, float_format='%.3f')
        print('\t', str(file_num - i), ' finished')

if __name__ == "__main__":
    s = StructuralNetworkGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                                   output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                                   hop=1)
    s.get_structural_network_all_time(worker=10)
