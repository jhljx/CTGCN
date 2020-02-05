import numpy as np
import pandas as pd
import os, multiprocessing, time
from numpy import random
import scipy.sparse as sp
from RWTGCN.utils import check_and_make_path, build_graph, read_edgelist_from_dataframe
from RWTGCN.preprocessing.helper import random_walk


class TensorGenerator:
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    walk_time: int
    walk_length: int
    prob: float

    # 这里的孤点(在nodelist而不在edgelist中的一定不会有游走序列，所以就不再图里添加这些孤点了)
    def __init__(self, base_path, input_folder, output_folder, node_file, walk_time=100, walk_length=5, prob=0.5):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        self.walk_time = walk_time
        self.walk_length = walk_length
        self.prob = prob

        check_and_make_path(self.output_base_path)
        tem_dir = ['walk_tensor']
        for tem in tem_dir:
            check_and_make_path(os.path.join(self.output_base_path, tem))

    def generate_tensor(self, f_name, original_graph_path, structural_graph_path, weight=True):
        # print('f_name = ', f_name)
        # f_folder = f_name.split('.')[0]
        # output_dir_path = os.path.join(self.output_base_path, 'walk_tensor', f_folder)
        # check_and_make_path(output_dir_path)
        #
        # original_graph_dict = build_graph(original_graph_path, self.full_node_list)
        # structural_graph_dict = build_graph(structural_graph_path, self.full_node_list)
        # t1 = time.time()
        # random_walk(original_graph_dict, structural_graph_dict, self.full_node_list, output_dir_path,
        #             self.walk_length, self.walk_time, self.prob, weight)
        # t2 = time.time()
        # print('random walk time = ', t2 - t1, ' seconds!')

        f_folder = f_name.split('.')[0]
        output_dir_path = os.path.join(self.output_base_path, 'walk_tensor', f_folder)

        original_graph = read_edgelist_from_dataframe(original_graph_path, self.full_node_list)
        structural_graph = read_edgelist_from_dataframe(structural_graph_path, self.full_node_list)
        nodes = self.full_node_list.copy()
        original_graph_dict, structural_graph_dict = {}, {}
        # preprocessing
        for node in nodes:
            original_neighbors = list(original_graph.neighbors(node))
            original_weight = np.array([original_graph[node][i]['weight'] for i in original_neighbors])
            original_graph_dict[node] = {'neighbor': original_neighbors}
            original_graph_dict[node]['weight'] = original_weight / original_weight.sum()

            structural_neighbors = list(structural_graph.neighbors(node))
            structural_weight = np.array([structural_graph[node][i]['weight'] for i in structural_neighbors])
            structural_graph_dict[node] = {'neighbor': structural_neighbors}
            structural_graph_dict[node]['weight'] = structural_weight / structural_weight.sum()

        node_num = len(self.full_node_list)
        nid2idx_dict = dict(zip(self.full_node_list, np.arange(node_num).tolist()))

        spmat_list, node_count_list, all_count_list = [sp.lil_matrix(1, 1)], [[]], [-1]
        spmat_list += [sp.lil_matrix((node_num, node_num)) for i in range(self.walk_length)]
        node_count_list += [np.zeros(node_num, dtype=int).tolist() for i in range(self.walk_length)]
        all_count_list += np.zeros(self.walk_length, dtype=int).tolist()

        # random walk
        for iter in range(self.walk_time):
            for node in nodes:
                eps = 1e-8
                walk = [node]
                while len(walk) < self.walk_length + 1:
                    cur = walk[-1]
                    rd = random.random()
                    if rd <= self.prob + eps:
                        neighbors = original_graph_dict[cur]['neighbor']
                        weights = original_graph_dict[cur]['weight']
                    else:
                        neighbors = structural_graph_dict[cur]['neighbor']
                        weights = structural_graph_dict[cur]['weight']
                    if len(neighbors) == 0:
                        break
                    walk.append(random.choice(neighbors, p=weights) if weight else random.choice(neighbors))
                seq_len = len(walk)
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        step = j - i
                        left_idx = nid2idx_dict[walk[i]]
                        #print(j, walk[j])
                        right_idx = nid2idx_dict[walk[j]]
                        spmat = spmat_list[step]
                        node_count = node_count_list[step]

                        spmat[left_idx, right_idx] += 1
                        spmat[right_idx, left_idx] += 1
                        node_count[left_idx] += 1
                        node_count[right_idx] += 1
                        all_count_list[step] += 2
        # calculate PPMI values
        for i in range(1, self.walk_length + 1):
            spmat = spmat_list[i].tocoo()
            node_count = node_count_list[i]
            all_count = all_count_list[i]
            df_PPMI = pd.DataFrame({'row': spmat.row, 'col': spmat.col, 'data': spmat.data}, dtype=int)

            def calc_PPMI(series):
                res = np.log(series['data'] * all_count / (node_count[series['row']] * node_count[series['col']]))
                if res < 0:
                    return 0
                return res

            df_PPMI['data'] = df_PPMI.apply(calc_PPMI, axis=1)
            spmat = sp.coo_matrix((df_PPMI['data'], (df_PPMI['row'], df_PPMI['col'])), shape=(node_num, node_num))
            sp.save_npz(os.path.join(output_dir_path, str(i) + ".npz"), spmat)
        return

    def generate_tensor_all_time(self, worker=-1):
        print("all file(s) in folder transform to tensor...")
        f_list = os.listdir(self.input_base_path)

        if worker <= 0:
            for i, f_name in enumerate(f_list):
                original_graph_path = os.path.join(self.input_base_path, f_name)
                structural_graph_path = os.path.join(self.output_base_path, 'structural_network', f_name)
                t1 = time.time()
                self.generate_tensor(f_name, original_graph_path=original_graph_path,
                                     structural_graph_path=structural_graph_path)
                t2 = time.time()
                print('generate tensor time: ', t2 - t1, ' seconds!')
                break
        else:
            worker = min(os.cpu_count(), self.walk_length, worker)
            pool = multiprocessing.Pool(processes=worker)
            print("\t\tstart " + str(worker) + " worker(s)")
            for i, f_name in enumerate(f_list):
                original_graph_path = os.path.join(self.input_base_path, f_name)
                structural_graph_path = os.path.join(self.output_base_path, 'structural_network', f_name)
                pool.apply_async(self.generate_tensor, (f_name, original_graph_path, structural_graph_path))
            pool.close()
            pool.join()


if __name__ == "__main__":
    tg = TensorGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                         output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                         walk_time=100, walk_length=5, prob=0.5)
    tg.generate_tensor_all_time(worker=-1)