import time
from RWTGCN.preprocessing.structure_generation import StructuralNetworkGenerator
from RWTGCN.preprocessing.tensor_generation import TensorGenerator


class Processing:
    structure_generator: StructuralNetworkGenerator
    tensor_generator: TensorGenerator

    def __init__(self, base_path, input_folder, output_folder, node_file, hop=1, ratio=1.0, max_cnt=10, min_sim=0.5,
                 walk_time=100, walk_length=5, p=0.5):
        self.structure_generator = StructuralNetworkGenerator(base_path, input_folder, output_folder, node_file,
                                                              hop=hop, ratio=ratio, max_cnt=max_cnt, min_sim=min_sim)
        self.tensor_generator = TensorGenerator(base_path, input_folder, output_folder, node_file,
                                                walk_time=walk_time, walk_length=walk_length, p=p)
        return

    def run(self, worker=-1):
        print('start generate subgraph!')
        t1 = time.time()
        self.structure_generator.prepare_subgraph_data_folder(worker=worker)
        t2 = time.time()
        print('subgraph sampling time: ', t2 - t1, ' seconds.')
        print('start generate structural network!')
        self.structure_generator.get_structural_network_all_time(worker=worker)
        t3 = time.time()
        print('structural network generation time: ', t3 - t2, ' seconds.')
        print('start generate tensor!')
        self.tensor_generator.generate_tensor_all_time(worker=worker)
        t4 = time.time()
        print('tensor generation time: ', t4 - t3, ' seconds.')

if __name__ == '__main__':
    processing = Processing(base_path="..\\..\\data\\email-eu", input_folder="1.format",
                            output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                            hop=2, ratio=1, max_cnt=50, min_sim=0.5,
                            walk_time=100, walk_length=5, p=0.5)
    processing.run(worker=2)
