import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import numpy.linalg as la


def loss(m1, m2):
    return la.norm(m1-m2, ord='fro')


def SVD(adj_matrix, c):
    c_rows = int(np.ceil(c*adj_matrix.shape[0]))
    L, M, R = np.linalg.svd(adj_matrix)
    A = np.dot(L[:,:c_rows] * M[:c_rows], R[:c_rows, :])

    return A


def ColibriS(adj_matrix, c, epsilon=0.5):
    A = adj_matrix
    c_rows = int(np.ceil(c*A.shape[0]))

    A_A = A**2
    p_x = A_A.sum(axis=0)/A_A.sum()
    I = np.random.choice(A.shape[1], size=(c_rows,), p=p_x)
    L = A[:, I[0]].reshape((A.shape[0], 1))
    M = (1/(L.transpose() @ L)).reshape((1,1))

    for k in range(1, c_rows):
        res = A[:, I[k]] - L @ M @ L.transpose() @ A[:, I[k]]
        if la.norm(res) < epsilon*la.norm(A[:, I[k]]):
            continue
        else:
            delta = la.norm(res) ** 2
            y = (M @ L.transpose() @ A[:, I[k]]).reshape((M.shape[0],1))
            M_lu = M + y @ y.transpose() / (delta ** 2)
            M_u = np.concatenate((M_lu, -y/delta), axis=1)
            M_d = np.concatenate((-y.transpose()/delta, np.array(1/delta).reshape((1,1))), axis=1)
            M = np.concatenate((M_u, M_d), axis=0)
            L = np.concatenate((L, A[:, I[k]].reshape(A.shape[0], 1)), axis=1)

    R = L.transpose() @ A
    A_reconstruct = L @ M @ R
    adj_matrix_reconstruct = A_reconstruct

    return L, M, R, adj_matrix_reconstruct
