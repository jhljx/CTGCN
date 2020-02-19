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
        print('start generate tensor!')
        self.tensor_generator.generate_tensor_all_time(worker=worker)
        t3 = time.time()
        print('tensor generation time: ', t3 - t2, ' seconds.')

def main():
    # only random walk on original graph(data for GCN static embedding)
    # prob = 1
    t1 = time.time()
    print('start gcn preprocessing!')
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/facebook/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    processing = Processing(base_path=base_path, origin_folder=origin_folder, structure_folder='', walk_pair_folder = 'gcn_walk_pairs',
                            node_freq_folder='gcn_node_freq', walk_tensor_folder='', node_file=node_file,
                            hop=5, max_neighbor_num=50, walk_time=10, walk_length=5, prob=1)
    processing.run(worker=30)
    t2 = time.time()
    print('finish gcn preprocessing! total cost time:', t2 - t1, ' seconds!')

    t1 = time.time()
    print('start mrgcn preprocessing!')
    # random walk on original graph(data for MRGCN static embedding
    # prob = 1
    processing = Processing(base_path=base_path, origin_folder=origin_folder, structure_folder='', walk_pair_folder='mrgcn_walk_pairs',
                            node_freq_folder='mrgcn_node_freq', walk_tensor_folder="mrgcn_walk_tensor", node_file=node_file,
                            hop=5, max_neighbor_num=20, walk_time=10, walk_length=5, prob=1)
    processing.run(worker=30)
    t2 = time.time()
    print('finish mrgcn preprocessing! total cost time:', t2 - t1, ' seconds!')

    t1 = time.time()
    #random walk on both original graph and structural graph for RWTGCN dynamic embedding
    # 0 < prob < 1
    processing = Processing(base_path=base_path, origin_folder=origin_folder, structure_folder='structural_network',
                            walk_pair_folder='walk_pairs', node_freq_folder='node_freq',
                            walk_tensor_folder="walk_tensor", node_file=node_file,
                            hop=5, max_neighbor_num=20, walk_time=10, walk_length=5, prob=0.5)
    processing.run(worker=30)
    t2 = time.time()
    print('finish rwtgcn preprocessing! total cost time:', t2 - t1, ' seconds!')

if __name__ == '__main__':
    main()
