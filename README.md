# CTGCN
K-Core based Temporal Graph Convolutional Network for Dynamic Graphs

**Requirements:**
- [Python](https://www.python.org/downloads/) >= 3.6
- [Numpy](https://github.com/numpy/numpy) >= 1.17.0
- [Scipy](https://github.com/scipy/scipy) >= 1.3.0
- [Pandas](https://github.com/pandas-dev/pandas) >= 0.25.1
- [Cython](https://github.com/cython/cython) >= 0.29.14
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) >= 0.21.2
- [Networkx](https://github.com/networkx/networkx) >= 2.3
- [Pytorch](https://github.com/pytorch/pytorch) == 1.5.1


https://pytorch-geometric.com/whl/

**Directory:**
    
    CTGCN/  
      CTGCN/  
        baseline/                    (implemented baselines, i.e. GCN, GAT, EvolveGCN, TIMERS)  
        evaluation/                  (evaluation tasks, i.e. link prediction, node classification)  
        preprocessing/               (preprocessing tasks, i.e. k-core decomposition, random walk)  
        embedding.py                 (data loader and different kinds of embedding)  
        graph.py                     (dynamic graph generation for static graphs and scalability data generation)  
        layers.py                    (All layers used in CTGCN)  
        metrics.py                   (Loss function)  
        models.py                    (All models of CTGCN)  
        train.py                     (main file used to train different embedding methods)  
        utils.py                     (utility functions)  
      data/  
        facebook/  
          0. input/                  (raw data)  
          1. format/                 (formatted dynamic graph data)  
          2. embedding/              (embedding results)  
          nodes_set/                 (node list file)  
          CTGCN/                     (intermediate data, i.e. k-core data, random walk data)  
        enron/
          ......

**Commands:**
1. compile 'preprocessing/helper.pyx' file.

       cd preprocessing  
       python3 setup.py build_ext --inplace  

2. preprocess dynamic graphs to generate k-core data and random walk data.

       python3 preprocessing/__init__.py

3. train graph embedding methods

       python3 train.py

**Baselines:**
1. GCN, GAT, EvolveGCN, timers  
[https://github.com/jhljx/CTGCN/tree/master/CTGCN/baseline](https://github.com/jhljx/CTGCN/tree/master/CTGCN/baseline)
2. DeepWalk, node2vec, struc2vec   
[https://github.com/shenweichen/GraphEmbedding](https://github.com/shenweichen/GraphEmbedding)
3. DynGEM   
[https://github.com/palash1992/DynamicGEM/blob/master/dynamicgem/embedding/dynSDNE.py](https://github.com/palash1992/DynamicGEM/blob/master/dynamicgem/embedding/dynSDNE.py) 

# Notes
1. Origin graph file names must be timestamp format or integer number format(other wise when training dynamic embedding, sorted(f_list) may return a wrong order of files)
2. Weighted random walk are set as default in the `get_walk_info` function of 'CTGCN/preprocessing/walk_generation.py' file.
3. The original graph edge data doesn't need to have a reverse edge for each edge, because the graph read functions (`get_sp_adj_mat` and `get_nx_graph` functions in 'CTGCN/utils.py') will add reverse edges automatically. All graph data sets are read by `get_sp_adj_mat` and `get_nx_graph` functions.
4. The original graph file header must be 'from_id, to_id, weight', or you will modify the 'get_nx_graph' function of 'CTGCN/utils.py' file. `get_sp_adj_mat` don't care the concrete header name, as long as the first 2 columns are node indices. If the original graph file has only 2 columns,  `get_sp_adj_mat` function will set edge weights as 1 in the 3rd column. If the original graph file has 3 columns, `get_sp_adj_mat` function will set edge weights as values the 3rd column.

# Reference
- [K-Core based Temporal Graph Convolutional Network for Dynamic Graphs](https://arxiv.org/abs/2003.09902)
