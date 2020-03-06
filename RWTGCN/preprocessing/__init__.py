import time, os, sys
sys.path.append("..")
from RWTGCN.preprocessing.structure_generation import StructuralNetworkGenerator
from RWTGCN.preprocessing.tensor_generation import TensorGenerator

class Processing:
    def __init__(self, base_path, origin_folder, structure_folder, walk_pair_folder, node_freq_folder, walk_tensor_folder, node_file,
                 hop=5, max_neighbor_num=100, walk_time=100, walk_length=5, prob=0.5):
        if prob < 1:
            self.structure_generator = StructuralNetworkGenerator(base_path, origin_folder, structure_folder, node_file,
                                                                  hop=hop, max_neighbor_num=max_neighbor_num)
        else:
            self.structure_generator = None
        self.tensor_generator = TensorGenerator(base_path, origin_folder, structure_folder, walk_pair_folder, node_freq_folder, walk_tensor_folder, node_file,
                                                walk_time=walk_time, walk_length=walk_length, prob=prob)
        return

    def run(self, worker=-1):
        t1 = time.time()
        print('start generate structural network!')
        if self.structure_generator is not None:
            self.structure_generator.get_structural_network_all_time(worker=worker)
        t2 = time.time()
        print('structural network generation time: ', t2 - t1, ' seconds.')
        # print('start generate tensor!')
        # self.tensor_generator.generate_tensor_all_time(worker=worker)
        # t3 = time.time()
        # print('tensor generation time: ', t3 - t2, ' seconds.')

def main():
    worker = -1
    dataset = 'enron'

    # only random walk on original graph(data for GCN static embedding)
    # prob = 1

    # t1 = time.time()
    # print('start gcn preprocessing!')
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    # processing = Processing(base_path=base_path, origin_folder=origin_folder, structure_folder='', walk_pair_folder = 'evolvegcn_walk_pairs',
    #                         node_freq_folder='evolvegcn_node_freq', walk_tensor_folder='', node_file=node_file,
    #                         hop=5, max_neighbor_num=50, walk_time=50, walk_length=2, prob=1)
    # processing.run(worker=worker)
    # t2 = time.time()
    # print('finish gcn preprocessing! total cost time:', t2 - t1, ' seconds!')
    #
    # t1 = time.time()
    # print('start mrgcn preprocessing!')
    # for prob in [0]:
    #     # random walk on original graph(data for MRGCN static embedding
    #     # prob = 1
    #     processing = Processing(base_path=base_path, origin_folder=origin_folder, structure_folder='mrgcn_structural_network',
    #                             walk_pair_folder='mrgcn_walk_pairs_' + str(prob), node_freq_folder='mrgcn_node_freq_' + str(prob),
    #                             walk_tensor_folder="mrgcn_walk_tensor_" + str(prob), node_file=node_file,
    #                             hop=5, max_neighbor_num=30, walk_time=10, walk_length=5, prob=prob)
    #     processing.run(worker=worker)
    #     t2 = time.time()
    #     print('finish mrgcn preprocessing! total cost time:', t2 - t1, ' seconds!')

    t1 = time.time()
    #random walk on both original graph and structural graph for RWTGCN dynamic embedding
    # 0 < prob < 1
    prob_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    prob_list = [0]

    for prob in prob_list:
        processing = Processing(base_path=base_path, origin_folder=origin_folder, structure_folder='rwtgcn_structural_network',
                                walk_pair_folder='rwtgcn_walk_pairs_' + str(prob), node_freq_folder='rwtgcn_node_freq_' + str(prob),
                                walk_tensor_folder='rwtgcn_walk_tensor_' + str(prob), node_file=node_file,
                                hop=5, max_neighbor_num=-1, walk_time=10, walk_length=5, prob=prob)
        processing.run(worker=worker)
    t2 = time.time()
    print('finish rwtgcn preprocessing! total cost time:', t2 - t1, ' seconds!')

if __name__ == '__main__':
    main()
