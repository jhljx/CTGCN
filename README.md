# RWT-GCN
Random Walk based Temporal Graph Convolutional Network for Dynamic Graphs

# Notes
1. Origin graph file names must be timestamp format or integer number format(other wise when training dynamic embedding, sorted(f_list) may reture a wrong order of files)
2. Unweighted random walk are set as default in the 'get_tensor' function of 'RWTGCN/preprocessing/tensor_generation.py' file. But weighted random walk are also supported.
3. The original graph edge data don't need to have reverse edge for each edge, the procedure will add reverse edge in 'get_sp_adj_mat' / 'build_graph' function of 'RWTGCN/utils.py' file.
4. The original graph file header must be 'from_id, to_id, weight', or you will modify the 'build_graph' function of 'RWTGCN/utils.py' file.