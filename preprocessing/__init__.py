# coding: utf-8
import time
from preprocessing.structure_generation import StructureInfoGenerator
from preprocessing.walk_generation import WalkGenerator


# Pre-processing class
class Processing:
    walk_generator: WalkGenerator

    def __init__(self, base_path, origin_folder, core_folder, walk_pair_folder, node_freq_folder, node_file, walk_time=100, walk_length=5):
        if core_folder is not None:
            self.structure_generator = StructureInfoGenerator(base_path, origin_folder, core_folder, node_file)
        else:
            self.structure_generator = None
        self.walk_generator = WalkGenerator(base_path, origin_folder, walk_pair_folder, node_freq_folder, node_file, walk_time=walk_time, walk_length=walk_length)

    def run(self, worker=-1, generate_core=True, run_walk=True, sep='\t', weighted=True):
        if self.structure_generator is not None and generate_core:
            t1 = time.time()
            print('start generate k core graph!')
            self.structure_generator.get_kcore_graph_all_time(sep=sep, worker=worker)
            t2 = time.time()
            print('finish generate k core graph! cost time:', t2 - t1, 'seconds!')
        if run_walk:
            t1 = time.time()
            print('start generate walk info!')
            self.walk_generator.get_walk_info_all_time(worker=worker, sep=sep, weighted=weighted)
            t2 = time.time()
            print('finish generate walk info! cost time: ', t2 - t1, ' seconds.')


# preprocessing for all supported GNN methods
def preprocess(method, args):
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    core_folder = args.get('core_folder', None)
    node_file = args['node_file']
    walk_pair_folder = args['walk_pair_folder']
    node_freq_folder = args['node_freq_folder']
    file_sep = args['file_sep']
    generate_core = args.get('generate_core', False)
    run_walk = args.get('run_walk', True)
    weighted = args['weighted']
    walk_time = args['walk_time']
    walk_length = args['walk_length']
    worker = args['worker']

    t1 = time.time()
    processing = Processing(base_path=base_path, origin_folder=origin_folder, core_folder=core_folder, walk_pair_folder=walk_pair_folder, node_freq_folder=node_freq_folder,
                            node_file=node_file, walk_time=walk_time, walk_length=walk_length)
    processing.run(worker=worker, generate_core=generate_core, run_walk=run_walk, sep=file_sep, weighted=weighted)
    t2 = time.time()
    print('finish ' + method + ' preprocessing! total cost time:', t2 - t1, ' seconds!')
