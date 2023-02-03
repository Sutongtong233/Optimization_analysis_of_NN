import scipy.sparse as sp
import numpy as np
import random
import torch

def set_seed(init_seed):
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed) # 用于numpy的随机数
    random.seed(init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)

def mx_to_torch_sparse_tensor(mx):
    sparse_mx=sp.csr_matrix(mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)

def get_one_hot(label,class_num):
    label = np.asarray(label)
    N = len(label)
    label = label[np.newaxis,:].T 
    label = torch.LongTensor(label)
    y_one_hot = torch.zeros(N, class_num).scatter_(1, label, 1)
    return y_one_hot

def edge_to_adj(edge_index, N):
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                       shape=(N, N),  dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj


