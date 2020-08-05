# CTGCN
K-Core based Temporal Graph Convolutional Network for Dynamic Graphs

# CTGCN Requirements
- [Python](https://www.python.org/downloads/) >= 3.6
- [Numpy](https://github.com/numpy/numpy) >= 1.18.1
- [Pandas](https://github.com/pandas-dev/pandas) >= 1.0.5
- [Scipy](https://github.com/scipy/scipy) >= 1.5.1
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) >= 0.23.1
- [Networkx](https://github.com/networkx/networkx) >= 2.4
- [Pytorch](https://github.com/pytorch/pytorch) == 1.5.1

If you want to use baselines provided by this project, other python libraries are needed.
- torch-scatter == 2.0.5
- torch-sparse == 0.6.6
- torch-spline-conv == 1.2.0
- torch-cluster == 1.5.6
- [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) == 1.6.0

Some binaries of pytorch-geometric related libraries can be found in 
https://pytorch-geometric.com/whl/. Note that in this project, the NVIDIA-SMI version is 418.67 and the CUDA version is 10.1.

# Directory
    
    CTGCN/    
        baseline/                    (implemented baselines, i.e. GCN, GAT, P-GNN, EvolveGCN...)  
        config/                      (configuration files and configuration tutorial)
        data/                        (data sets)  
            enron/  
                0. input/                  (raw data)  
                1. format/                 (formatted dynamic graph data)  
                2. embedding/              (embedding results)  
                CTGCN/                     (intermediate data, i.e. k-core data, random walk data)
                nodes_set/                 (node list file)    
            facebook/
            ......
        evaluation/                  (evaluation tasks, i.e. link prediction, node classification)  
        preprocessing/               (preprocessing tasks, i.e. k-core decomposition, random walk)  
        embedding.py                 (data loader and different kinds of embedding)  
        graph.py                     (dynamic graph generation and scalability data generation)  
        layers.py                    (All layers used in CTGCN)  
        main.py                      (Main file of this project)
        metrics.py                   (Loss function)  
        models.py                    (All models of CTGCN)  
        train.py                     (main file used to train different embedding methods)  
        utils.py                     (utility functions)          

# Commands & Functions

We provide a docker file to help you build a docker environment. The docker commands of creating CTGCN containers are:

1. Creating a CTGCN CPU container

       docker run -it -v /home/xxx/CTGCN:/project -v /home/xxx/CTGCN/data:/data --name=CTGCN --memory=180G --cpus=35 ctgcn/ctgcn:v1 /bin/bash

2. Creating a CTGCN GPU container

       docker run -it -v /home/xxx/CTGCN:/project -v /home/xxx/CTGCN/data:/data --name=CTGCN_GPU --memory=180G --cpus=35 --runtime=nvidia ctgcn/ctgcn:v1 /bin/bash

The above docker commands are only examples. If you want to run CTGCN codes in a docker environment, you need to modify the file path, memory capacity, cpu thread number and docker image name in the above commands.

This project has several functions, including: **preprocessing**, **graph embedding**, **link prediction**, **node classification**, **edge classification** and **regular equivalence prediction**. Thus, the corresponding Python commands are:

1. **Preprocessing**: generate k-core subgraphs and perform random walk.

       python3 main.py --config=config/uci.json --task=preprocessing --method=CTGCN-C

2. **Graph Embedding**: perform graph embedding approaches on several dynamic graph data sets.

       python3 main.py --config=config/uci.json --task=embedding --method=CTGCN-C

3. **Link Prediction**: perform link prediction on several dynamic graph data sets to test the performance of graph embedding approaches.

       python3 main.py --config=config/uci.json --task=link_pred
   
4. **Node Classification**: perform node classification on several dynamic graph data sets to test the performance of graph embedding approaches. 

       python3 main.py --config=config/america_air.json --task=node_cls

5. **Edge Classification**: perform edge classification on several dynamic graph data sets to test the performance of graph embedding approaches. 

       python3 main.py --config=config/xxx.json --task=edge_cls

    Note that we don't have edge classification data sets, so this function is only left for your future usage. Please pay attention that the code of this function is not fully tested.

6. **Regular Equivalence Prediction**: perform regular equivalence prediction on several dynamic graph data sets to test the performance of graph embedding approaches. 

       python3 main.py --config=config/america_air.json --task=equ_pred

# Parameter Configurations

All other configuration parameters are saved in configuration files. For more detailed configuration information, please refer to [config/README.md](https://github.com/jhljx/CTGCN/tree/master/config).

# Baselines

We provide unified pytorch (or python) version of many graph embedding approaches in this project.

1. Graph Convolutional Network (GCN)　[\[paper\]](https://arxiv.org/abs/1609.02907)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/gcn.py)
2. Graph Attention Network (GAT)　[\[paper\]](https://arxiv.org/abs/1710.10903)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/gat.py)
3. Sample and Aggregate (GraphSAGE)　[\[paper\]](https://arxiv.org/abs/1706.02216)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/sage.py)
4. Graph Isomorphism Network (GIN)　[\[paper\]](https://arxiv.org/abs/1810.00826)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/gin.py)   
5. Position-aware Graph Neural Network (P-GNN)　[\[paper\]](https://arxiv.org/abs/1906.04817)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/pgnn.py)   
6. Graph Convolutional Recurrent Network (GCRN)　[\[paper\]](https://arxiv.org/abs/1612.07659)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/gcrn.py)   
7. Variational Graph Recurrent Network (VGRNN)　[\[paper\]](https://arxiv.org/abs/1908.09710)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/vgrnn.py)    
8. Evolving Graph Convolutional Network (EvolveGCN)　[\[paper\]](https://arxiv.org/abs/1902.10191)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/egcn.py)    
9. DynGEM　[\[paper\]](https://arxiv.org/abs/1805.11273)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/dynGEM.py)      
10. dyngraph2vec　[\[paper\]](https://arxiv.org/abs/1809.02657)
   
   - dynAE　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/dynAE.py)
   - dynRNN　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/dynRNN.py) 
   - dynAERNN　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/dynAERNN.py) 

11. Theoretically Instructed Maximum-Error-bounded Restart of SVD (TIMERS)　[\[paper\]](https://arxiv.org/abs/1711.09541)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/timers.py)   


# Notes
1. Origin graph file names must be timestamp format or integer number format, otherwise when training dynamic embedding, sorted(f_list) may return a wrong order of files.
2. Weighted random walk are set as default in the `get_walk_info` function of 'preprocessing/walk_generation.py' file.
3. The original graph edge data doesn't need to have a reverse edge for each edge, because the graph read functions (`get_sp_adj_mat` and `get_nx_graph` functions in 'utils.py') will add reverse edges automatically. All graph data sets are read by `get_sp_adj_mat` and `get_nx_graph` functions.
4. The original graph file header must be 'from_id, to_id, weight', or you will modify the 'get_nx_graph' function of 'utils.py' file. `get_sp_adj_mat` don't care the concrete header name, as long as the first 2 columns are node indices. If the original graph file has only 2 columns,  `get_sp_adj_mat` function will set edge weights as 1 in the 3rd column. If the original graph file has 3 columns, `get_sp_adj_mat` function will set edge weights as values the 3rd column.

# Reference
- [K-Core based Temporal Graph Convolutional Network for Dynamic Graphs](https://arxiv.org/abs/2003.09902)
