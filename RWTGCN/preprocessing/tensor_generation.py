import numpy as np
import pandas as pd
import os, multiprocessing, time
from scipy import sparse
from RWTGCN.utils import check_and_make_path
from RWTGCN.preprocessing.random_walk import HybridRandomWalk


class TensorGenerator:
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    random_walk: HybridRandomWalk
    walk_length: int

    # 这里的孤点(在nodelist而不在edgelist中的一定不会有游走序列，所以就不再图里添加这些孤点了)
    def __init__(self, base_path, input_folder, output_folder, node_file, walk_time=100, walk_length=5, p=0.5):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        self.random_walk = HybridRandomWalk(base_path, input_folder, output_folder, node_file,
                                            walk_time=walk_time, walk_length=walk_length, p=p)
        self.walk_length = walk_length

        check_and_make_path(self.output_base_path)
        tem_dir = ['walk_tensor']
        for tem in tem_dir:
            check_and_make_path(os.path.join(self.output_base_path, tem))

    def generate_tensor_all_time(self, worker=-1):
        print("all file(s) in folder transform to tensor...")
        f_list = os.listdir(self.input_base_path)
        length = len(f_list)
        for i, f_name in enumerate(f_list):
            print("\t", length - i, "file(s) left")
            original_graph_path = os.path.join(self.input_base_path, f_name)
            structural_graph_path = os.path.join(self.output_base_path, 'structural_network', f_name)
            walks = self.random_walk.get_walk_sequences(original_graph_path=original_graph_path,
                                                        structural_graph_path=structural_graph_path)
            self.generate_tensor(walks, f_name, worker=worker)

    # 多worker，大文件时候好像pool回收会有问题，卡了好久
    def generate_tensor(self, walks, f_name, worker=-1):
        print("\t\tget distribution tensor...")
        f_folder = f_name.split('.')[0]
        output_folder = os.path.join(self.output_base_path, 'walk_tensor', f_folder)
        if os.path.exists(output_folder) and len(os.listdir(output_folder)) == self.walk_length:
            print('\t\t', f_name, "is processed")
            return
        check_and_make_path(output_folder)
        if worker <= 0:
            for i in range(1, self.walk_length + 1):
                self.single_layer(walks, output_folder, i)
        else:
            worker = min(os.cpu_count(), self.walk_length, worker)
            pool = multiprocessing.Pool(processes=worker)
            print("\t\tstart " + str(worker) + " worker(s)")
            for i in range(1, self.walk_length + 1):
                pool.apply_async(self.single_layer, (walks, output_folder, i))
            pool.close()
            pool.join()
        print("\t\tdistribution tensor got")

    def single_layer(self, walks, output_dir_path, i):
        print("\t\t\tinterval:", str(i))
        t1 = time.time()
        node_num = len(self.full_node_list)
        nid2idx_dict = dict(zip(self.full_node_list, np.arange(node_num).tolist()))

        from scipy.sparse import lil_matrix
        spmat = lil_matrix((node_num, node_num))
        for walk in walks:
            left = 0
            right = left + i
            length = len(walk)
            while right < length:
                left_idx = nid2idx_dict[walk[left]]
                right_idx = nid2idx_dict[walk[right]]
                spmat[left_idx, right_idx] += 1
                spmat[right_idx, left_idx] += 1
                left += 1
                right += 1
        spmat = spmat.tocoo()
        sparse.save_npz(os.path.join(output_dir_path, str(i) + ".npz"), spmat)
        t2 = time.time()
        print("\t\t\tinterval:", str(i), "finished, use", t2 - t1, "seconds")


if __name__ == "__main__":
    tg = TensorGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                         output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv")
    tg.generate_tensor_all_time(worker=-1)
