# coding: utf-8
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds, eigs
from numpy import random
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, diags, find
from numpy.linalg import pinv
from itertools import product
import os
from utils import get_sp_adj_mat


# TIMERS: Error-Bounded SVD Restart on Dynamic Networks. For more information, please refer to https://arxiv.org/abs/1711.09541
# We refer to the TIMERS matlab source code in https://github.com/ZW-ZHANG/TIMERS and implement a python version of TIMERS
# Author: jhljx
# Email: jhljx8918@gmail.com


def get_sp_delta_adj_mat(A_pre, file_path, node2idx_dict, sep='\t'):
    N = len(list(node2idx_dict.keys()))
    A_cur = get_sp_adj_mat(file_path, node2idx_dict, sep=sep)
    delta_A = lil_matrix((N, N))
    pre_row, pre_col, pre_value = find(A_pre)
    cur_row, cur_col, cur_values = find(A_cur)
    pre_edge_num = pre_row.size
    cur_edge_num = cur_row.size

    pre_dict = dict(zip(zip(pre_row, pre_col), pre_value))
    cur_dict = dict(zip(zip(cur_row, cur_col), cur_values))
    for idx in range(pre_edge_num):
        edge = (pre_row[idx], pre_col[idx])
        if edge in cur_dict:
            delta_A[pre_row[idx], pre_col[idx]] = cur_dict[edge] - pre_dict[edge]
        else:
            delta_A[pre_row[idx], pre_col[idx]] = - pre_dict[edge]

    for idx in range(cur_edge_num):
        edge = (cur_row[idx], cur_col[idx])
        if edge in pre_dict:
            delta_A[cur_row[idx], cur_col[idx]] = cur_dict[edge] - pre_dict[edge]
        else:
            delta_A[cur_row[idx], cur_col[idx]] = cur_dict[edge]

    delta_A = delta_A.tocsr()
    return delta_A

# Sim is the N x N sparse similarity matrix
# U is the N x K left embedding vector
# V is the N x K right embedding vector
# returns ||S -U * V^T||_F ^2
def Obj(Sim, U, V):
    # A trick to reduce space complexity to O(M)
    K = U.shape[1]
    PS_u = np.dot(U.transpose(), U)  # matrix multiplication
    PS_v = np.dot(V.transpose(), V)

    # PS_u PS_v are the K x K matrix, pre-calculated sum for embedding vector
    # PS_u(i, j) = sum_k=1^N U(k, i)U(k, j)
    # PS_u(i, j) = sum_k=1^N V(k, i)V(k, j)
    temp_row, temp_col, temp_value = find(Sim)
    # calculate first term
    # print('len = ', len(temp_row))
    L = np.multiply(temp_value, temp_value).sum()
    # print('L = ', L)

    # calculate second term
    M = len(temp_value)
    # print('M = ', M)
    # separated into k iteration to avoid memory overflow
    for i in range(1, K + 1):
        start_index = np.floor((i - 1) * M / K + 1).astype(np.int)
        end_index = np.floor(i * M / K).astype(np.int)
        temp_inner = np.multiply(U[temp_row[start_index - 1: end_index], :],
                                 V[temp_col[start_index - 1: end_index], :]).sum(axis=1)
        L = L - 2 * (np.multiply(temp_value[start_index - 1:end_index], temp_inner).sum())

    # calculate third term
    L = L + np.multiply(PS_u, PS_v).sum()
    return L


def Obj_SimChange(S_ori, S_add, U, V, loss_ori):
    # S_ori is n x n symmetric sparse original similarity matrix
    # Note, S_ori is last similarity matrix, not static one
    # S_add is n x n symmetric sparse new similarity, may overlap with S_ori
    # U / V: n x k left / right embedding vectors
    # K: embedding diemsnion
    # loss_ori: original loss, i.e. | | S_ori - U * V ^ T | | _F ^ 2
    # return new loss, i.e | | S_ori + S_add - U * V ^ T | | _F ^ 2

    N, K = U.shape
    temp_row, temp_col, temp_value = find(S_add)
    M = len(temp_value)  # temp_value type: ndarray
    S_ori_lil = S_ori.tolil()
    temp_old_value = S_ori_lil[temp_row, temp_col].transpose()  # type is sparse matrix, col_num == 1
    loss_new = loss_ori

    # avoid memory overflow
    for i in range(1, K + 1):
        start_index = np.floor((i - 1) * M / K + 1).astype(np.int)
        end_index = np.floor(i * M / K).astype(np.int)
        Ua = U[temp_row[start_index - 1:end_index], :]
        Va = V[temp_col[start_index - 1: end_index], :]
        temp_inner = np.multiply(Ua, Va).sum(axis=1)  # ndarray

        temp_old_value_arr = temp_old_value[start_index - 1:end_index].toarray().flatten()  # array column num == 1, make it from 2D-array to a 1-D array
        temp_value_arr = temp_value[start_index - 1:end_index].flatten()
        loss_new = loss_new - np.power((temp_old_value_arr - temp_inner), 2).sum()
        loss_new = loss_new + np.power((temp_old_value_arr + temp_value_arr - temp_inner), 2).sum()

    return loss_new


def Random_Com(N, M, p, seed, c_num, c_size, c_prob):
    # N: number of nodes
    # M: number of random edges(approximately)
    # p: proportion in static matrix, A
    # seed: random seed
    # c_num: number of communities
    # c_size: approximate community size
    # c_prob: approximate edge forming probability
    # returns random graph + community appearing
    # create random network, undirected, unweighted

    random.seed(seed)
    temp_row = random.randint(N, size=np.around(M / 2).astype(np.int))
    temp_col = random.randint(N, size=np.around(M / 2).astype(np.int))
    temp_data = np.ones(shape=np.around(M / 2).astype(np.int))

    # temp is a sparse matrix
    temp = csr_matrix((temp_data, (temp_row, temp_col)), shape=(N, N))

    # print(temp)

    temp = temp - diags(temp.diagonal(k=0), 0)

    temp = temp + temp.transpose()  # undirected

    temp_row, temp_col, _ = find(temp)  # get rid of multiple edges

    temp_choose = (temp_row > temp_col)

    temp_row = temp_row[temp_choose]
    temp_col = temp_col[temp_choose]

    del temp_choose

    # randomly generate order
    temp_order = random.permutation(len(temp_row))
    temp_row = temp_row[temp_order]
    temp_col = temp_col[temp_order]

    del temp_order

    # split into static and dynamic network
    temp_num = np.around(p * len(temp_row)).astype(np.int)

    A_row = temp_row[:temp_num]
    A_col = temp_col[:temp_num]
    A_data = np.ones(shape=len(A_row))
    A = csr_matrix((A_data, (A_row, A_col)), shape=(N, N))

    A = A + A.transpose()
    E = np.hstack((temp_row[temp_num:].reshape(-1, 1), temp_col[temp_num:].reshape(-1, 1)))

    # randomly generate a timestamp
    TimeStamp = np.sort(random.rand(E.shape[0], 1))

    # create community edge
    # temp: store all existings edges to ensure no overlapping
    c_add = []
    for i in range(1, c_num + 1):
        c_node = random.choice(N, c_size, replace=False)
        c_temp = lil_matrix((N, N))

        c_node_x, c_node_y = list(zip(*list(product(c_node, c_node))))
        c_temp[c_node_x, c_node_y] = 1
        c_temp = c_temp.tocsr()

        c_temp = c_temp - c_temp.multiply((temp > 0))
        temp_row, temp_col, _ = find(c_temp)
        temp_choose = (temp_row < temp_col)
        temp_row = temp_row[temp_choose]
        temp_col = temp_col[temp_choose]
        temp_choose = (random.rand(len(temp_row)) <= c_prob)
        temp_row = temp_row[temp_choose]
        temp_col = temp_col[temp_choose]
        temp_order = random.permutation(len(temp_row))
        temp_row = temp_row[temp_order]
        temp_col = temp_col[temp_order]
        if len(c_add) == 0:
            c_add = np.hstack((temp_row.reshape(-1, 1), temp_col.reshape(-1, 1)))
            # print('c_add shape: ', c_add.shape)
        else:
            c_add = np.vstack((c_add, np.hstack((temp_row.reshape(-1, 1), temp_col.reshape(-1, 1)))))
        temp = temp.tolil()
        # print('type: ', type(temp))
        temp[temp_row, temp_col] = 1
        temp[temp_col, temp_row] = 1
        temp = temp.tocsr()

    temp_insert = np.around(random.rand() * E.shape[0] * 0.6).astype(np.int)  # avoid too late change
    # create a simulated time
    t_add = np.sort(random.rand(len(c_add), 1))
    t_add = TimeStamp[temp_insert] + np.multiply(t_add, (TimeStamp[temp_insert + 1] - TimeStamp[temp_insert]))
    E = np.vstack((E[:temp_insert, :], np.vstack((c_add, E[temp_insert:, :]))))
    TimeStamp = np.vstack((TimeStamp[:temp_insert], np.vstack((t_add, TimeStamp[temp_insert:]))))

    print('Node number:', str(N),
          '; Edge number ', str((A > 0).sum()),
          '; New edge number:', str(2 * len(E)),
          '(Community:', str(2 * len(c_add)), ')')
    return [A, E, TimeStamp]


def RefineBound(S_ori, S_add, Loss_ori, K):
    # S_ori is n x n symmetric sparse original similarity matrix
    # S_add is n x n symmetric sparse new similarity, may overlap with S_ori
    # Loss_ori is the value or lower bound of loss for S_ori
    # K is embedding dimension
    # Return a lower bound of loss for S_ori + S_add by matrix perturbation inequality

    # In short, Loss_Bound = Loss_ori + trace_change(S * S ^ T) - eigs(delta(S * S ^ T), K)
    # Check our paper for detail:
    # Zhang, Ziwei, et al. "TIMERS: Error-Bounded SVD Restart on Dynamic Networks".AAAI, 2018.

    # Calculate trace change
    # S_overlap = (S_add != 0).multiply(S_ori)
    # S_temp = S_add + S_overlap
    # trace_change = (S_temp.multiply(S_temp)).sum() - (S_overlap.multiply(S_overlap)).sum()
    # del S_overlap
    # del S_temp

    S_temp = S_add + S_ori
    trace_change = (S_temp.dot(S_temp)).diagonal().sum() - (S_ori.dot(S_ori)).diagonal().sum()
    del S_temp

    # Calculate eigenvalues sum of delta(S * S ^ T)
    # Notice we only need to deal with non - zero rows / columns
    S_temp = S_ori.dot(S_add)
    S_temp = S_temp + S_temp.transpose() + S_add.dot(S_add)
    # _, S_choose, _ = find(S_temp.sum(axis=0))
    # S_temp = S_temp.tocsr()
    # S_temp = S_temp[S_choose, :].tocsc()
    # S_temp = S_temp[:, S_choose]
    # S_temp = S_temp.tocsr().astype(np.float)
    # del S_choose
    # note eigs return largest absolute value, instead of largest
    eigen_num = min(int(np.around(2 * K)), S_temp.shape[0])
    temp_eigs, _ = eigs(S_temp, eigen_num)  # return eigen value array and eigen vector array
    temp_eigs = temp_eigs.real
    temp_eigs = temp_eigs[temp_eigs >= 0]
    temp_eigs = np.sort(temp_eigs)[::-1]

    if (len(temp_eigs) >= K):
        eigen_sum = temp_eigs[:K].sum()
    else:  # if doesn't calculate enough, add another inequality
        temp_l = len(temp_eigs)
        eigen_sum = temp_eigs.sum() + temp_eigs[temp_l - 1] * (K - temp_l)

    # Calculate loss lower bound
    #print('loss origin: ', Loss_ori, ', trace change: ', trace_change, ', eigen sum: ', eigen_sum)
    Loss_Bound = Loss_ori + trace_change - eigen_sum
    return Loss_Bound


def TRIP(Old_U, Old_S, Old_V, Delta):
    # update using TRIP method
    # reference: Chen Chen, and Hanghang Tong. "Fast eigen-functions tracking on dynamic graphs." SDM, 2015.
    N, K = Old_U.shape
    Delta_A = Delta.copy()
    # solve eigenvalue and eigenvectors from SVD, denote as L, X
    Old_X = Old_U.copy()
    for i in range(1, K + 1):  # unify the sign
        temp_i = np.argmax(np.abs(Old_X[:, i - 1]))  # 返回max的位置
        if (Old_X[temp_i, i - 1] < 0):
            Old_X[:, i - 1] = -Old_X[:, i - 1]

    [temp_v, temp_i] = np.max(Old_U, axis=0), np.argmax(Old_U, axis=0)
    y_idx = [i for i in range(K)]

    temp_sign = np.sign(np.multiply(temp_v, (Old_V[temp_i, y_idx])))  # use maximum absolute value to determine sign
    Old_L = np.multiply(np.diag(Old_S).transpose(), temp_sign)  # 1 x k eigenvalues
    del temp_v, temp_i, temp_sign

    # calculate sum term

    Delta_A = Delta_A.tocsc()
    sp_Old_x_t = csr_matrix(Old_X.transpose())
    sp_Old_x = csc_matrix(Old_X)

    temp_sum = sp_Old_x_t.dot(Delta_A).dot(sp_Old_x)
    # calculate eigenvalues change
    Delta_L = temp_sum.diagonal(k=0)
    # calculate eigenvectors change
    Delta_X = np.zeros(shape=(N, K))
    for i in range(1, K + 1):
        # print(np.ones(shape=K).dot(Old_L[i - 1] + Delta_L[i - 1]) - Old_L)
        temp_D = np.diag(np.ones(shape=K).dot(Old_L[i - 1] + Delta_L[i - 1]) - Old_L)
        sp_piv = csr_matrix(pinv(temp_D - temp_sum))
        temp_sum = temp_sum.tocsc()
        temp_alpha = sp_piv.dot(temp_sum[:, i - 1])
        # print(temp_alpha.shape)
        sp_Old_x = csr_matrix(Old_X)
        Delta_X[:, i - 1] = sp_Old_x.dot(temp_alpha).toarray().flatten()

    # return updated result
    New_U = (Old_X + Delta_X)
    for i in range(1, K + 1):
        New_U[:, i - 1] = np.divide(New_U[:, i - 1], (np.sqrt(New_U[:, i - 1].transpose().dot(New_U[:, i - 1]))))

    New_S = np.diag(np.abs(Old_L + Delta_L))
    New_V = New_U.dot(np.diag(np.sign(Old_L + Delta_L)))
    return New_U, New_S, New_V


def timers(nodes_file, input_base_path, output_base_path, Theta=0.17, dim=128, sep='\t', Update=True):
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    # nodes set
    nodes_set = pd.read_csv(nodes_file, names=['node'])
    full_node_list = nodes_set['node'].tolist()
    N = len(full_node_list)
    print('node num: ', N)
    node2idx_dict = dict(zip(full_node_list, np.arange(N)))
    # base graph adj list
    f_list = sorted(os.listdir(input_base_path))
    f0 = os.path.join(input_base_path, f_list[0])
    A = get_sp_adj_mat(f0, node2idx_dict, sep=sep)

    # begin TIMERS
    K = dim
    time_slice = len(f_list) - 1

    # Store all results
    U = [[] for i in range(time_slice + 10)]
    S = [[] for i in range(time_slice + 10)]
    V = [[] for i in range(time_slice + 10)]
    Loss_store = np.zeros(shape=time_slice + 10)  # store loss for each time stamp
    Loss_bound = np.zeros(shape=time_slice + 10)  # store loss bound for each time stamp
    run_times = 0  # store how many rerun times
    Run_t = np.zeros(shape=time_slice + 10)  # store which timeslice re - run


    # Calculate Static Solution
    u, s, vt = svds(A, K)
    s = np.diag(s)
    v = vt.transpose()
    U[0], S[0], V[0] = u, s, v
    U_cur = np.dot(U[0], np.sqrt(S[0]))
    V_cur = np.dot(V[0], np.sqrt(S[0]))
    Loss_store[0] = Obj(A, U_cur, V_cur)
    Loss_bound[0] = Loss_store[0]
    output_data = np.hstack((U_cur, V_cur))
    assert output_data.shape[0] == N
    result = pd.DataFrame(data=output_data, index=full_node_list, columns=range(2 * dim))
    result.to_csv(os.path.join(output_base_path, f_list[0]), sep=sep)
    print('time = 1, loss = ', Loss_store[0], ', loss_bound=', Loss_bound[0])
    # Store some useful variable
    Sim = A.copy()
    S_cum = A.copy()  # store cumulated similarity matrix
    del A

    S_perturb = csr_matrix((N, N))  # store cumulated perturbation from last rerun
    loss_rerun = Loss_store[0]  # store objective function of last rerun
    for i in range(1, time_slice + 1):

        # create the change in adjacency matrix
        fn = os.path.join(input_base_path, f_list[i])
        S_add = get_sp_delta_adj_mat(S_cum, fn, node2idx_dict, sep=sep)
        S_perturb = S_perturb + S_add
        if Update:
            # Some Updating Function Here
            [U[i], S[i], V[i]] = TRIP(U[i - 1], S[i - 1], V[i - 1], S_add)
            # We use TRIP as an example, while other variants are permitted (as discussed in the paper)
            # Note that TRIP doesn't ensure smaller loss value
            U_cur = np.dot(U[i], np.sqrt(S[i]))
            V_cur = np.dot(V[i], np.sqrt(S[i]))
            Loss_store[i] = Obj(S_cum + S_add, U_cur, V_cur)
        else:
            Loss_store[i] = Obj_SimChange(S_cum, S_add, U_cur, V_cur, Loss_store[i - 1])

        Loss_bound[i] = RefineBound(Sim, S_perturb, loss_rerun, K)
        S_cum = S_cum + S_add
        print('time = ', i + 1, ', loss = ', Loss_store[i], ', loss_bound=', Loss_bound[i])
        if (Loss_store[i] >= (1 + Theta) * Loss_bound[i]):
            print('Begin rerun at time stamp:', str(i + 1))
            Sim = S_cum.copy()
            S_perturb = csr_matrix((N, N))
            run_times = run_times + 1
            Run_t[run_times] = i

            u, s, vt = svds(Sim, K)
            s = np.diag(s)
            v = vt.transpose()

            U[i], S[i], V[i] = u, s, v

            U_cur = np.dot(U[i], np.sqrt(S[i]))
            V_cur = np.dot(V[i], np.sqrt(S[i]))
            loss_rerun = Obj(Sim, U_cur, V_cur)
            Loss_store[i] = loss_rerun
            Loss_bound[i] = loss_rerun
        print('time = ', i + 1, ', loss = ', Loss_store[i], ', loss_bound=', Loss_bound[i])
        assert U_cur.shape[0] == V_cur.shape[0]
        assert U_cur.shape[1] == V_cur.shape[1]
        output_data = np.hstack((U_cur, V_cur))
        assert output_data.shape[0] == N
        result = pd.DataFrame(data=output_data, index=full_node_list, columns=range(2 * dim))
        result.to_csv(os.path.join(output_base_path, f_list[i]), sep=sep)
    del S_cum, S_perturb, Sim
    del loss_rerun
    del U_cur, V_cur


def timers_embedding(args):
    # common params
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    embedding_folder = args['embed_folder']
    node_file = args['node_file']
    file_sep = args['file_sep']
    embed_dim = args['embed_dim']

    # timers model params
    theta = args['theta']

    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_base_path = os.path.abspath(os.path.join(base_path, embedding_folder))
    node_file_path = os.path.abspath(os.path.join(base_path, node_file))

    timers(node_file_path, origin_base_path, embedding_base_path, Theta=theta, dim=embed_dim // 2, sep=file_sep, Update=True)
