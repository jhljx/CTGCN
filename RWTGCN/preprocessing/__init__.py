from RWTGCN.preprocessing.structure_generation import StructuralNetworkGenerator
from RWTGCN.preprocessing.tensor_generation import TensorGenerator


class Processing:
    structure_generator: StructuralNetworkGenerator
    tensor_generator: TensorGenerator

    def __init__(self, base_path, input_folder, output_folder, node_file, hop=1, ratio=1, max_cnt=10, min_sim=0.5,
                 walk_time=100, walk_length=5, p=0.5):
        self.structure_generator = StructuralNetworkGenerator(base_path, input_folder, output_folder, node_file,
                                                              hop=hop, ratio=ratio, max_cnt=max_cnt, min_sim=min_sim)
        self.tensor_generator = TensorGenerator(base_path, input_folder, output_folder, node_file,
                                                walk_time=walk_time, walk_length=walk_length, p=p)
        return

    def run(self, worker=-1):
        self.structure_generator.prepare_subgraph_data_folder(worker=worker)
        self.structure_generator.get_structural_network_all_time(worker=worker)
        self.tensor_generator.generate_tensor_all_time(worker=worker)


if __name__ == '__main__':
    processing = Processing(base_path="..\\data\\email-eu", input_folder="1.format",
                                  output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                                  hop=1, ratio=1, max_cnt=10, min_sim=0.5,
                                  walk_time=100, walk_length=10, p=0.5)
    # processing.run(worker=2)
