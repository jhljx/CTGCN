import time, sys
sys.path.append("..")
from RWTGCN.preprocessing.structure_generation import StructuralNetworkGenerator
from RWTGCN.preprocessing.tensor_generation import TensorGenerator


class Processing:
    structure_generator: StructuralNetworkGenerator
    tensor_generator: TensorGenerator

    def __init__(self, base_path, input_folder, output_folder, node_file, hop=5, max_neighbor_num=100,
                 walk_time=100, walk_length=5, prob=0.5):
        self.structure_generator = StructuralNetworkGenerator(base_path, input_folder, output_folder, node_file,
                                                              hop=hop, max_neighbor_num=max_neighbor_num)
        self.tensor_generator = TensorGenerator(base_path, input_folder, output_folder, node_file,
                                                walk_time=walk_time, walk_length=walk_length, prob=prob)
        return

    def run(self, worker=-1):
        t1 = time.time()
        print('start generate structural network!')
        self.structure_generator.get_structural_network_all_time(worker=worker)
        t2 = time.time()
        print('structural network generation time: ', t2 - t1, ' seconds.')
        print('start generate tensor!')
        self.tensor_generator.generate_tensor_all_time(worker=worker)
        t3 = time.time()
        print('tensor generation time: ', t3 - t2, ' seconds.')

if __name__ == '__main__':
    processing = Processing(base_path="../../data/facebook", input_folder="1.format",
                            output_folder="RWT-GCN", node_file="nodes_set/nodes.csv",
                            hop=5, max_neighbor_num=100, walk_time=10, walk_length=5, prob=0.5)
    processing.run(worker=30)
    # processing = Processing(base_path="..\\..\\data\\email-eu", input_folder="1.format",
    #                         output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
    #                         hop=5, max_neighbor_num=50, walk_time=100, walk_length=5, prob=0.5)
    # processing.run(worker=4)
