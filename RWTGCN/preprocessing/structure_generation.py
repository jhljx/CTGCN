import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, multiprocessing, random
import sys, time
sys.path.append("..")
from RWTGCN.utils import check_and_make_path, wl_transform

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
        print('start wl transform!')
        new_labels, cluster_dict = wl_transform(spadj, labels, cluster=True)
        cluster_len_dict = {}
        for cluster_type, cluster_list in cluster_dict.items():
            cluster_len_dict[cluster_type] = len(cluster_list)

        color_arr = new_labels.reshape(-1, 1)
        for i in range(1, self.hop):
            new_labels = wl_transform(spadj, new_labels, cluster=False)
            color_arr = np.hstack((color_arr, new_labels.reshape(-1, 1)))
        print('finish wl transform!')
        print('start get structural neighbor!')
        # print('node num = ', color_arr.shape[0])
        t1 = time.time()
        try:
            import RWTGCN.preprocessing.helper as helper
            helper.get_structural_neighbors(color_arr, output_file, cluster_dict,
                                            cluster_len_dict, idx2nid_dict, self.max_neighbor_num)
        except:
            import RWTGCN.utils as utils
            print('use util get structural neighbors!')
            utils.get_structural_neighbors(color_arr, output_file, cluster_dict,
                                               cluster_len_dict, idx2nid_dict, self.max_neighbor_num)
        t2 = time.time()
        print('finish get structural neighbor!')
        print('cost time: ', t2 - t1, ' seconds!')

if __name__ == "__main__":
    s = StructuralNetworkGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                                   output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                                   hop=1)
    s.get_structural_network_all_time(worker=-1)
