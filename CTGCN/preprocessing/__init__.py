import time, os, sys
sys.path.append("..")
from CTGCN.preprocessing.structure_generation import StructureInfoGenerator
from CTGCN.preprocessing.walk_generation import WalkGenerator

class Processing:
    def __init__(self, base_path, origin_folder, core_folder, walk_pair_folder, node_freq_folder, node_file, walk_time=100, walk_length=5):
        if core_folder is not None:
            self.structure_generator = StructureInfoGenerator(base_path, origin_folder, core_folder, node_file)
        else:
            self.structure_generator = None
        self.walk_generator = WalkGenerator(base_path, origin_folder, walk_pair_folder, node_freq_folder, node_file, walk_time=walk_time, walk_length=walk_length)
        return

    def run(self, worker=-1, calc_structure=True):
        if self.structure_generator is not None and calc_structure:
            t1 = time.time()
            print('start generate k core graph!')
            self.structure_generator.get_kcore_graph_all_time(worker=worker)
            t2 = time.time()
            print('finish generate k core graph! cost time:', t2 - t1, 'seconds!')

        t1 = time.time()
        print('start generate walk info!')
        self.walk_generator.get_walk_info_all_time(worker=worker)
        t2 = time.time()
        print('finish generate walk info! cost time: ', t2 - t1, ' seconds.')

def gcn_process(dataset, worker=-1):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    t1 = time.time()
    processing = Processing(base_path=base_path, origin_folder=origin_folder, core_folder=None,
                            walk_pair_folder='gcn_walk_pairs', node_freq_folder='gcn_node_freq', node_file=node_file,
                            walk_time=10, walk_length=5)
    processing.run(worker=worker)
    t2 = time.time()
    print('finish gcn preprocessing! total cost time:', t2 - t1, ' seconds!')

def gat_process(dataset, worker=-1):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    t1 = time.time()
    processing = Processing(base_path=base_path, origin_folder=origin_folder, core_folder=None,
                            walk_pair_folder='gat_walk_pairs', node_freq_folder='gat_node_freq', node_file=node_file,
                            walk_time=10, walk_length=5)
    processing.run(worker=worker)
    t2 = time.time()
    print('finish gat preprocessing! total cost time:', t2 - t1, ' seconds!')

def evolvegcn_process(dataset, worker=-1):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    t1 = time.time()
    processing = Processing(base_path=base_path, origin_folder=origin_folder, core_folder=None,
                            walk_pair_folder='evolvegcn_walk_pairs', node_freq_folder='evolvegcn_node_freq', node_file=node_file,
                            walk_time=10, walk_length=5)
    processing.run(worker=worker)
    t2 = time.time()
    print('finish evolvegcn preprocessing! total cost time:', t2 - t1, ' seconds!')

def cgcn_process(dataset, worker=-1, calc_structure=True):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    t1 = time.time()
    processing = Processing(base_path=base_path, origin_folder=origin_folder, core_folder='cgcn_cores',
                            walk_pair_folder='cgcn_walk_pairs', node_freq_folder='cgcn_node_freq', node_file=node_file,
                            walk_time=10, walk_length=5)
    processing.run(worker=worker, calc_structure=calc_structure)
    t2 = time.time()
    print('finish cgcn preprocessing! total cost time:', t2 - t1, ' seconds!')

def ctgcn_process(dataset, worker=-1, calc_structure=True):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    t1 = time.time()
    processing = Processing(base_path=base_path, origin_folder=origin_folder, core_folder='ctgcn_cores',
                            walk_pair_folder='ctgcn_walk_pairs', node_freq_folder='ctgcn_node_freq', node_file=node_file,
                            walk_time=10, walk_length=5)
    processing.run(worker=worker, calc_structure=calc_structure)
    t2 = time.time()
    print('finish ctgcn preprocessing! total cost time:', t2 - t1, ' seconds!')

if __name__ == '__main__':
    dataset = 'facebook'
    worker = 30
    #gcn_process(dataset=dataset, worker=worker)
    #gat_process(dataset=dataset, worker=worker)
    #evolvegcn_process(dataset=dataset, worker=worker)
    # cgcn_process(dataset=dataset, worker=worker, calc_structure=True)
    ctgcn_process(dataset=dataset, worker=worker, calc_structure=True)
