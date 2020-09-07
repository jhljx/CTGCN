# Configuration

Here we provide program configurations in this project. This project have several kinds of functions in graph representation learning, including **static graph embedding**, **dynamic graph embedding**, **link prediction**, **node classification** and **graph centrality prediction**. 

Then configurations for each function will be introduced. We also give the detailed parameter usage guide.

## Preprocessing
The parameters used in the **preprocessing** task is shown as follows. Note that If the graph embedding model doesn't use the unsupervised **negative sampling** loss (i.e. DynGEM, dyngraph2vec, TIMERS, etc), then the preprocessing process is **unnecessary**!

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path` |
| core_folder | str |  k-core subgraphs folder path relative to `base_path` <br> (**optional**, value: can be null) |
| node_file | str | node file path relative to `base_path` |
| walk_pair_folder | str | random walk node co-occurring pairs file folder path relative to `base_path` |
| node_freq_folder | str | node frequency file folder path relative to `base_path` |
| file_sep | str |  file separator for all files (i.e. '\t') |
| generate_core | bool | whether to generate k-core subgraph files or not <br> (**optional**, value: true or false) |
| run_walk | bool | whether to run random walk or not <br> (**optional**, value: true or false) |
| weighted | bool | weighted graph or not, then this will determine weighted random walk or not |
| walk_time | int | random walk times for each node |
| walk_length | int | random walk length for each node |
| worker | int | CPU multiprocessing thread number <br> (default: -1, means don't use multiprocessing) |


Note that **optional** parameters can be omitted in the configuration file if this optional parameter is not needed. CGCN-C and CGCN-S share the same `core_folder`. So when running the preprocessing task for these methods, you only need to generate output files once.

We also provide a bool parameter `generate_core` to control the generation of k-core subgraphs. K-core subgraphs will be generated only and if only `core_file` is not null and `generate_core` is true. If you only want to execute random walk in the preprocessing task, you can set `generate_core` as false for CGCN-C, CGCN-S, CTGCN-C and CTGCN-S methods (because their `core_file` is not null), while you can set `core_file` as null for other methods as they don't need k-core subgraphs.

## Graph Embedding

The graph embedding task contains the execution of several graph embedding methods, including static graph embedding end dynamic graph embedding methods. The parameters of all supported graph embedding methods are shown below. 

For a DynamicGEM based graph embedding method, its parameter set is the union of **Common parameters**, **DynamicGEM common parameters** and its model parameters (i.e. **DynGEM parameters**).

For a GNN based graph embedding method, its parameter set is the union of **Common parameters**, **GNN common parameters**, **Learning parameters**(optional) and its model parameters (i.e. **GCN parameters**).

For TIMERS, its parameter set is **TIMERS parameters**.

### Common parameters
Common parameters are input parameters used by all supported graph embedding methods (except TIMERS) in the CTGCN library, including DynGEM related methods(DynGEM and dyngraph2vec) and GNN methods.

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path` |
| core_folder | str | k-core subgraph folder path relative to `base_pah` <br> (**optional**, value: can be null) |
| embed_folder | str | embedding folder path relative to `base_path` |
| model_folder | str | graph embedding model folder path relative to `base_path` | 
| model_file | str | model file name <br> (value: **can be null**, then model won't be saved!) |
| node_file | str | node file path relative to `base_path` |
| file_sep | str | file separator for all files (i.e. '\t') |
| start_idx | int | start timestamp index of dynamic graphs  |
| end_idx | int | end timestamp index of dynamic graphs <br> (default: -1, means the last timestamp, it supports negative values) |
| duration | int | The timestamp length of the input |
| embed_dim | int | node embedding dimension |
| use_cuda | bool | whether or not to use GPU for calculation |
| thread_num | int | the thread number for training if CPU is used |
| epoch | int | training epoch number of a graph embedding model |
| lr | float | learning rate of the optimizer |
| batch_size | int | batch size of a data batch <br> (value: > 0, no other constraints) |
| load_model | bool | whether or not to load graph embedding model for training |
| shuffle | bool | whether or not to shuffle the data for training |
| export | bool | whether or not to save node embedding files for each timestamp |
| record_time | bool | whether or not to record the running time |

Note that **optional** parameters can be omitted in the configuration file if this optional parameter is not needed.  Moreover, the timestamp range is a closed interval \[start_idx, end_idx\]. The program will add 1 to `end_idx`, making the range into a left-closed interval. 

Usually, the `duration` parameter is greater than 1 for dynamic graph embedding; for static graph embedding, `duration` is set as 1.


### DynamicGEM parameters
DynamicGEM parameters include input parameters for DynGEM and dyngraph2vec(DyAE, DyRNN, DynAERNN) methods.

#### Common parameters

Here we provide common parameters used by both DynGEM and dyngraph2vec.

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| beta | float |  reconstruction penalty |
| nu1 | float | relative weight hyper-parameter of L1-regularization loss |
| nu2 | float | relative weight hyper-parameter of L2-regularization loss |
| bias | bool | whether or not to enable bias for model layers |

#### DynGEM parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| n_units | list |  hidden layer dimension list of its encoder |
| alpha | float | relative weight hyper-parameter of its first order proximity loss |

#### DynAE parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| n_units | list |  hidden layer dimension list of its encoder |
| look_back | int | history timestamp length used for capturing time-evolving patterns |


#### DynRNN parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| n_units | list |  hidden layer dimension list of its LSTM encoder |
| look_back | int | history timestamp length used for capturing time-evolving patterns |


#### DynAERNN parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| ae_units | list |  hidden layer dimension list of its AE encoder |
| rnn_units | list |  hidden layer dimension list of its LSTM encoder |
| look_back | int | history timestamp length used for capturing time-evolving patterns |


### GNN parameters
Here we provide input parameters of all supported GNN methods. 

#### Common parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| nfeature_folder | str | node feature folder path relative to `base_path` <br> (**optional**, value: can be null) |
| learning_type | str | learning type of gnns <br> (value: 'U-neg', 'U-own', 'S-node', 'S-edge', 'S-link-st', 'S-link-dy') |
| hid_dim | int | dimension of hidden layers in a graph embedding model |
| weight_decay | int | weight decay of the optimizer |

Note that `learning_type` is a parameter for choosing different learning strategies. Moreover, in all configuration files, we don't include `nfeature_folder` parameter, because all data sets used in this project don't have node features. But if you want to run programs on graph data sets with node features, you can add the `nfeature_folder` parameter in your configuration file.

#### Learning parameters
Learning parameters are parameters related to unsupervised learning or supervised learning. These parameters are all **optional parameters**. 

| **Parameter** | **Type** | **Description** | Usage |
|:----:|:----:| :----: | :----: |
| walk_pair_folder | str | random walk co-occurring node pairs folder path relative to `base_path` | Unsupervised Learning
| node_freq_folder | str | random walk node frequency folder path relative to `base_path` | Unsupervised Learning
| neg_num | int | negative sample number |  Unsupervised Learning |
| Q | float | penalty weight of negative sampling term in negative sampling loss| Unsupervised Learning |
| nlabel_folder | str | node label folder path relative to `base_path` | Node Classification |
| elabel_folder | str | edge label folder path relative to `base_path` | Edge Classification |
| cls_file | str | classifier file name <br> (file saved in `model_folder`, only file name is needed) | Supervised Classification |
| cls_hid_dim | int | dimension of hidden layers in the classifier model |  Supervised Classification |
| cls_layer_num | int | layer number in the classifier model | Supervised Classification |
| cls_bias | bool | whether or not to enable bias for classifier layers | Supervised Classification |
| cls_activate_type | str | activation function type in the classifier model <br> (value: 'L' or 'N', linear or non-linear) | Supervised Classification |
| train_ratio | float | the ratio of training nodes(edges) to all nodes(edges) in each graph | Supervised Learning |
| val_ratio | float | the ratio of validation nodes(edges) to all nodes(edges) in each graph | Supervised Learning |
| test_ratio | float | the ratio of test nodes(edges) to all nodes(edges) in each graph | Supervised Learning |

Note that `walk_pair_folder`, `node_freq_folder`, `neg_num` and `Q` are used for unsupervised negative sampling loss. Parameters from `cls_file` to `cls_activate_type` are used for building the node classifier or edge classifier. `train_ratio`, `val_ratio` and `test_ratio` are used for data split in supervised node classification, edge classification and link prediction. 

Moreover, the input_dim of a classifier is `embed_dim`, and the output_dim of a classifier is the unique label number.

Here we introduce how to change different learning strategies.

- Unsupervised learning with negative sampling loss (`learning_type` = 'U-neg')
- Unsupervised learning with its own loss (`learning_type` = 'U-own')
- Supervised learning for node classification (`learning_type` = 'S-node')
- Supervised learning for edge classification (`learning_type` = 'S-edge')
- Supervised learning for static(or dynamic) link prediction (`learning_type` = 'S-link-st' or 'S-link-dy')

Note that **the difference between 'S-link-st' and 'S-link-dy' is **:

- 'S-link-st' is designed for static supervised link prediction based model training, while 'S-link-dy' is designed for dynamic supervised link prediction based model training  
- 'S-link-st' uses the current embedding matrix to predict edge labels of the current timestamp, while 'S-link-dy' uses the previous embedding matrix to predict edge labels of the current timestamp. So 'S-link-dy' needs 'duration' >= 2 and 'end_idx' - 'start_idx' >= 1 for both static and dynamic gnn methods.
- In `embedding.py` files, the only difference between 'S-link-st' and 'S-link-dy' strategies is `edge_label_list = edge_label_list[1:]` and `embedding_list = embedding_list[:-1]`, which makes the previous embedding matrix match the current edge label matrix.

#### GCN parameters
Original GCN parameters

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |

Note that here GCN has two graph convolution layers.

#### TgGCN parameters
We also provide another GCN version which is implemented by pytorch_geometric library.

Pytorch-Geometric GCN(TgGCN) parameters

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| feature_pre | bool | whether or not to add a linear layer before all GCN layers |
| feature_dim | int | output dimension of the added linear layer |
| layer_num | int | GCN layer num |
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |


#### GAT parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |
| alpha | float | negative slope angle of LeakyReLU used in GAT |
| head_num | int | number of multi-head attention |

Note that here GAT has two graph attention layers, head_num=1, negative_slop=0.2.

#### TgGAT parameters
We also provide another GAT version which is implemented by pytorch_geometric library.

Pytorch-Geometric GAT(TgGAT) parameters

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| feature_pre | bool | whether or not to add a linear layer before all GAT layers |
| feature_dim | int | output dimension of the added linear layer |
| layer_num | int | GAT layer num |
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |

Note that here GAT uses some default parameters of 'tg.nn.GATConv', So the `heads` parameter is 1, `negative_slope` parameter is 0.2.

#### SAGE parameters

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| num_sample | int | the number of sampled neighborhood for all nodes <br> (value: can be null or positive number) |
| pooling_type | str | neighborhood aggregation type <br> (value: 'sum', 'average', 'max') |
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |


#### TgSAGE parameters

We also provide another SAGE version which is implemented by pytorch_geometric library.

Pytorch-Geometric SAGE(TgSAGE) parameters

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| feature_pre | bool | whether or not to add a linear layer before all SAGE layers |
| feature_dim | int | output dimension of the added linear layer |
| layer_num | int | SAGE layer num |
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |

Note that here GraphSAGE uses some default parameters of 'tg.nn.SAGEConv', So the `pool` parameter is 'mean'.

#### GIN parameters

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| layer_num | int | GIN layer num |
| mlp_layer_num | int | layer num in each MLP |
| learn_eps | bool | whether to learn epsilon to distinguish center nodes from neighboring nodes or not |
| pooling_type | str | neighbor aggregation type <br> (value: 'sum', 'average', 'max') |
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |


#### TgGIN parameters

We also provide another GIN version which is implemented by pytorch_geometric library.

Pytorch-Geometric GIN(TgGIN) parameters


| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| feature_pre | bool | whether or not to add a linear layer before all GIN layers |
| feature_dim | int | output dimension of the added linear layer |
| layer_num | int | GIN layer num |
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |

Note that here GIN uses some default parameters of 'tg.nn.GINCov'. So the `eps` parameter is 0, and `train_eps` parameter is False. And both parameters are not included in the above parameter list.


#### P-GNN parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| feature_pre | bool | whether or not to add a linear layer before all P-GNNN layers |
| feature_dim | int | output dimension of the added linear layer |
| layer_num | int | P-GNN layer num |
| dropout | float | dropout rate (range: \[0, 1\]) |
| bias | bool | whether or not to enable bias for model layers |
| approximate | int | pairwised shortest path cutoff <br> (default: -1, means exact shortest path calculation) |


#### GCRN parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| dropout | float | dropout rate (range: \[0, 1\]) |
| rnn_type | str | rnn type to identify different rnns used in GCRN <br> (value: 'GRU' or 'LSTM') |
| bias | bool | whether or not to enable bias for model layers |

Note that GCRN uses GCN as its backbone, so the default GCN layer number is 2.

#### VGRNN parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| rnn_layer_num | int | rnn layer number in VGRNN |
| eps | float | eps parameter used in its KL component of the VAE loss |
| conv_type | str | graph convolution type <br> (value: 'GCN', 'SAGE', 'GIN') |
| bias | bool | whether or not to enable bias for model layers |

Note that VGRNN uses GRU to capture temporal features.

#### EvolveGCN parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| init_type | str | degree-based input feature initialization type <br> (value: 'gaussian', 'one-hot', 'adj', 'combine')  |
| std | float | std of the gaussian distribution if `init_type` is 'gaussian' <br> (**optional**) |
| model_type | str | EvolveGCN model type <br> (value: 'EGCNH', 'EGCNO')

Note that in the EvolveGCN paper, it only uses one-hot node features as input.

#### CGCN-C, CGCN-S, CTGCN-C, CTGCN-S parameters
| **Parameter** | **Type** | **Description** | **Usage** |
|:----:|:----:| :----: | :----: |
| max_core | int | number of k-core subgraphs for each dynamic graph <br> (default: -1, means all k-core subgraphs in each graph are used) | Common  |
| trans_layer_num | int | feature transformation layer number | Common |
| diffusion_layer_num | int | core-based diffusion (or CGCN) layer number | Common |
| init_type | str | degree-based input feature initialization type <br> (**optional**, value: 'gaussian', 'one-hot', 'adj', 'combine')  | CGCN-S, CTGCN-S |
| std | float | std of the gaussian distribution if `init_type` is 'gaussian' <br> (**optional**) |  CGCN-S, CTGCN-S |
| model_type | str | model type to identify different versions of the model <br> (value: 'C' or 'S', means C-version or S-version | Common |
| rnn_type | str | rnn type to identify different rnns used in CGCN <br> (value: 'GRU' or 'LSTM') | Common |
| trans_activate_type | str | activation function type of feature transformation layers <br> (value: 'L' or 'N', means 'linear' or 'non-linear') | Common |
| bias | bool | whether or not to enable bias for model layers | Common |

Note that if `max_core` is greater than the k-core number of a graph, then all k-core subgraphs are used in CGCN layers.


### TIMERS parameters
The parameters used in the TIMERS method is as follows:

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path` |
| embed_folder | str | embedding folder path relative to `base_path` |
| node_file | str | node file path relative to `base_path` |
| file_sep | str |  file separator for all files (i.e. '\t') |
| embed_dim | int | node embedding dimension |
| theta | float | threshold for restarting SVD <br> (default: 0.17, range: \[0, 1\])

## Link Prediction
The parameters used in the **link prediction** task is shown as follows:

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path` |
| embed_folder | str |  embedding folder path relative to `base_path` |
| node_file | str | node file path relative to `base_path` |
| lp_edge_folder | str | the link prediction edge data folder path relative to `base_path`|
| lp_res_folder | str | the link prediction results folder path relative to `base_path` |
| file_sep | str |  file separator for all files (i.e. '\t') |
| start_idx | int | The start index of repeating link prediction |
| rep_num | int |  The repetition time of link prediction |
| train_ratio  | float | the ratio of training edges to all edges for each graph |
| val_ratio | float | the ratio of validation edges to all edges for each graph  |
| test_ratio | float | the ratio of test edges to all edges for each graph |
| do_lp | bool | whether to generate lp data and lp results for `rep_num` times |
| generate | bool | whether to generate the split labeled edge data |
| aggregate | bool | whether to aggregate `rep_num` times of lp results into one file|
| method_list | list | a list of graph embedding methods used to make link prediction |
| c_list | list | a list of C parameter used in the sklearn LogisticRegression model |
| measure_list | list | a list of evaluation metrics <br> (support 4 edge features: 'avg', 'had', 'l1', 'l2') |
| max_iter | int | maximum iteration number in the sklearn LogisticRegression model |
| worker | int | CPU multiprocessing thread number <br> (default: -1, means don't use multiprocessing) |

Note that we use for loop to generate `rep_num` times random link prediction edge data and its corresponding link prediction results. Then after running `rep_num` times of link prediction, all `rep_num` times of link prediction results will be aggregated.

We provide flexibility to run link prediction task. If `do_lp` is true and `aggregate` is false, then the program will only calculate link prediction results for `rep_num` times. If `do_lp` is false and`aggregate` is true, then only the aggregation function will be executed. Another useful bool parameter is `generate`. If you want to test the performance of graph embedding methods later, you can set `generate=false` to used the link prediction data which was generated before.

## Node Classification
The parameters used in the **node classification** task is shown as follows:

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path` |
| embed_folder | str |  embedding folder path relative to `base_path` |
| node_file | str | node file path relative to `base_path` |
| nlabel_folder | str | node label folder path relative to `base_path` |
| nodecls_data_folder | str | node data folder path relative to `base_path`<br> (train, val and test) |
| nodecls_res_folder | str | node classification results data |
| file_sep | str | separator of all files | 
| start_idx | int | The start index of repeating node classification |
| rep_num | int | The repetition time of node classification |
| train_ratio | float | the ratio of training nodes to all nodes in each graph |
| val_ratio | float | the ratio of validation nodes to all nodes in each graph |
| test_ratio | float | the ratio of test nodes to all nodes in each graph |
| do_nodecls | bool | whether to generate node cls data and node cls results for `rep_num` times |
| generate | bool | whether to generate the split labeled node data |
| aggregate | bool | whether to aggregate `rep_num` times of nodecls results into one file |
| method_list | list | a list of graph embedding methods used to make node classification |
| c_list | list | a list of C parameter used in the sklearn LogisticRegression model |
| max_iter | int | maximum iteration number in the sklearn LogisticRegression model |
| worker | int | CPU multiprocessing thread number <br> (default: -1, means don't use multiprocessing) |

## Edge Classification
The parameters used in the **edge classification** task is shown as follows:

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path` |
| embed_folder | str |  embedding folder path relative to `base_path` |
| node_file | str | node file path relative to `base_path` |
| elabel_folder | str | edge label folder path relative to `base_path` |
| edgecls_data_folder | str | edge data folder path relative to `base_path`<br> (train, val and test) |
| edgecls_res_folder | str | edge classification results data |
| file_sep | str | separator of all files | 
| start_idx | int | The start index of repeating edge classification |
| rep_num | int | The repetition time of edge classification |
| train_ratio | float | the ratio of training edges to all edges in each graph |
| val_ratio | float | the ratio of validation edges to all edges in each graph |
| test_ratio | float | the ratio of test edges to all edges in each graph |
| do_edgecls | bool | whether to generate edge cls data and edge cls results for `rep_num` times |
| generate | bool | whether to generate the split labeled edge data |
| aggregate | bool | whether to aggregate `rep_num` times of edgecls results into one file |
| method_list | list | a list of graph embedding methods used to make edge classification |
| c_list | list | a list of C parameter used in the sklearn LogisticRegression model |
| max_iter | int | maximum iteration number in the sklearn LogisticRegression model |
| worker | int | CPU multiprocessing thread number <br> (default: -1, means don't use multiprocessing) |

## Graph Centrality Prediction

The parameters used in the **graph centrality prediction** task is shown as follows:

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path`|
| embed_folder | str |  embedding folder path relative to `base_path` |
| node_file | str | node file path relative to `base_path` |
| centrality_data_folder | str | graph centrality data folder path relative to `base_path` |
| centrality_res_folder | str |  graph centrality prediction results folder path relative to `base_path` |
| file_sep | str |  file separator for all files (i.e. '\t') |
| generate | bool | whether to generate centrality data |
| method_list | list | a list of graph embedding methods used to make graph centrality prediction |
| alpha_list | list | alpha parameter list used in the sklearn RidgeRegression model |
| split_fold | int | cross validation fold number |
| worker | int | CPU multiprocessing thread number <br> (default: -1, means don't use multiprocessing) |

Note that graph files in the origin folder have already been generated. Each graph file have 3 columns separated by '\t'. The first row of each graph file is its header.