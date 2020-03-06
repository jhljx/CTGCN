import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, multiprocessing, random
import sys, time
sys.path.append("..")
from RWTGCN.utils import check_and_make_path, wl_transform, get_sp_adj_mat, build_graph

class StructuralNetworkGenerator:
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    node_num: int
    hop: int
    max_neighbor_num: int
    max_nxt_num: int

    def __init__(self, base_path, input_folder, output_folder, node_file, hop=5, max_neighbor_num=-1, max_nxt_num=-1):
        self.base_path = base_path
        self.input_base_path = os.path.abspath(os.path.join(base_path, input_folder))
        self.output_base_path = os.path.abspath(os.path.join(base_path, output_folder))

        node_path = os.path.abspath(os.path.join(base_path, node_file))
        nodes_set = pd.read_csv(node_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)
        self.hop = hop
        self.max_neighbor_num = max_neighbor_num
        self.max_nxt_num = max_nxt_num

        check_and_make_path(self.output_base_path)

    def get_structural_network(self, input_file, output_file, file_num, idx):
        import RWTGCN.preprocessing.helper as helper
        print('\t', str(file_num - idx), ' file(s) left')
        if os.path.exists(output_file):
            print('\t', output_file, "exist")
            print('\t', str(file_num - idx), ' finished')
            return
        t1 = time.time()
        # spadj must be symmetric matrix(input file don't need to have reverse edge)
        spadj = get_sp_adj_mat(input_file, self.full_node_list)
        # degrees = spadj.sum(axis=1).astype(np.int)
        # pd.DataFrame(degrees).to_csv('degrees.csv', sep=',', index=False)

        neighbor_dict, core_arr = build_graph(input_file, self.full_node_list, core=True)
        # print('core arr shape: ', core_arr.shape)
        print(np.max(core_arr))
        return
        # pd.DataFrame(core_arr).to_csv('core_num.csv', sep=',', index=False)
        cluster_dict = dict()
        for i, core_num in enumerate(core_arr):
            if core_num not in cluster_dict:
                cluster_dict[core_num] = [i]
            else:
                cluster_dict[core_num].append(i)
        # print(list(cluster_dict.keys()))

        structure_edge_dict = dict()
        labels = np.ones(self.node_num, dtype=np.int)
        max_label = np.max(labels)

        print('start wl transform!')
        # helper.calc_structural_weight(cluster_dict, neighbor_dict, core_arr, labels, max_label, structure_edge_dict, self.max_neighbor_num, self.max_nxt_num, 0)
        # print('finish!')
        spadj_list = [spadj]
        for i in range(self.hop):
            spadj_list.append(spadj_list[-1].dot(spadj))
            labels = wl_transform(spadj, labels, max_label)
            max_label = np.max(labels)
        print('finish wl transform!')
        print('max_label: ', max_label)
        helper.calc_structural_weight(cluster_dict, spadj_list, core_arr, labels, max_label, structure_edge_dict, self.max_neighbor_num, self.max_nxt_num, 5)
        structural_edge_list = []
        for edge, weight in structure_edge_dict.items():
            structural_edge_list.append([self.full_node_list[edge[0]], self.full_node_list[edge[1]], weight])
        df_structural_edges = pd.DataFrame(structural_edge_list, columns=['from_id', 'to_id', 'weight'])
        print('edge num: ', df_structural_edges.shape[0])
        df_structural_edges.to_csv(output_file, sep='\t', index=False, header=True, float_format='%.3f')

        t2 = time.time()
        print('cost time: ', t2 - t1, ' seconds!')

    def get_structural_network_all_time(self, worker=-1):
        print("getting all timestamps structural network adjacent...")

        f_list = os.listdir(self.input_base_path)
        length = len(f_list)
        if worker <= 0:
            for i, f_name in enumerate(f_list):
                self.get_structural_network(
                    input_file=os.path.join(self.input_base_path, f_name),
                    output_file=os.path.join(self.output_base_path, f_name),
                    file_num=length, idx=i)
        else:
            worker = min(worker, length, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for i, f_name in enumerate(f_list):
                pool.apply_async(self.get_structural_network, (
                    os.path.join(self.input_base_path, f_name),
                    os.path.join(self.output_base_path, f_name), length, i,))
            pool.close()
            pool.join()
        print("got it...")

if __name__ == "__main__":
    s = StructuralNetworkGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                                   output_folder="RWT-GCN\\structural_network", node_file="nodes_set\\nodes.csv",
                                   hop=1)
    s.get_structural_network_all_time(worker=-1)
