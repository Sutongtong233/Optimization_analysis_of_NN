import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, FakeDataset
from torch import optim
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("..")
sys.path.append(".")

import configargparse
from GNN import *
from utils import *
   
import matplotlib.pyplot as plt
import copy

def generate_attributed_graph(args):
    data = FakeDataset(num_graphs=1, avg_num_nodes=args.N, avg_degree=args.avg_degree, num_channels=args.D, num_classes=args.C).data
    t_features = data.x
    t_labels = data.y
    t_edge_index = data.edge_index
    N = t_features.shape[0]
    adj = edge_to_adj(data.edge_index, N) # sp.coo_matrix
    adj = normalize(adj)
    adj = adj + adj.T
    adj[adj > 1] = 1

    return t_features, t_labels, t_edge_index, adj, data
    

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(1001):
        optimizer.zero_grad()
        output = F.log_softmax(model(t_features, sparse_mx_to_torch_sparse_tensor(adj)), dim=1)
        loss = F.nll_loss(output, t_labels)
        acc = accuracy(output, t_labels)
        loss.backward()
        optimizer.step()
        if (epoch % 100 == 0):
            print("Epoch: {}".format(epoch + 1),
                 "loss: {:.4f}".format(loss.item()),
                 "acc: {:.4f}".format(acc.item()))
    _, predicted = torch.max(output.data, 1)          
    return acc, output, predicted

def create_random_direction(W):
    """
        W: [W1, W2]

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
   
    weights = W
    direction = [torch.randn(w.size()) for w in weights]
    # normalize_directions_for_weights(direction, weights, norm, ignore)
    for d, w in zip(direction, weights):
        d.mul_(w.norm()/(d.norm() + 1e-10))
   
    return direction



def rand_2D_plot(model_opt, alpha, beta):
    # W_opt: [W0_opt, W1_opt]
    W_0, W_1, _ = get_model_params(model_opt)
    # model_inter_0 = copy.deepcopy(model_opt)
    # model_inter_1 = copy.deepcopy(model_opt)
    model_inter = copy.deepcopy(model_opt)

    direction = create_random_direction([W_0, W_1])
    direction_0 = direction[0]
    direction_1 = direction[1]
    for name, parameters in model_inter.named_parameters():
        if name == 'gcn1.weight': 
            parameters.data = alpha*direction_0 + (1-alpha)*W_0
        if name == 'gcn2.weight': 
            parameters.data = beta*direction_1 + (1-beta)*W_1
    output = F.log_softmax(model_inter(t_features, sparse_mx_to_torch_sparse_tensor(adj)), dim=1)
    loss = F.nll_loss(output, t_labels)
    return torch.round(loss, decimals=4).detach().numpy()



def get_model_params(model):
    W_list=[]
    for name, parameters in model.named_parameters():
        W_list.append(parameters)
    W_0 = W_list[0]
    W_1 = W_list[1]
    W = torch.mm(W_0, W_1)
    return W_0.clone(), W_1.clone(), W.clone()



if __name__=='__main__':
    
    parser = configargparse.ArgumentParser()
    parser.add_argument('--N', help='expected number of nodes', default=500)
    parser.add_argument('--D', help='nodes feature dimension', default=128)
    parser.add_argument('--C', help='class number', default=2)
    parser.add_argument('--hidden_dim', default=16)
    parser.add_argument('--avg_degree', help='average degree of the graph', default=4)
    parser.add_argument('--seed', default=42)

    parser.add_argument('--lamb', default=1)

    args = parser.parse_args()

    

    random.seed(args.seed) # control N
    torch.manual_seed(args.seed) # control features

    t_features, t_labels, t_edge_index, adj, data = generate_attributed_graph(args)
    N = t_features.shape[0]
    print(f"###### total {N} nodes #####")
    print(f"###### features: {t_features[0].max()} #####")
    A = torch.tensor(adj.todense())
    # two random initialization of model
    model_0 = SGC(input_dim=args.D, hidden_dim=args.hidden_dim, output_dim=args.C)

    # two corresponding optimized model
    acc_before, output_before, predict_before = train(model_0)
    print(f"accuracy for model: {acc_before}")


  
    # Exp 3: bilinear interpolation 
    
    n_inter = 10
    loss_ls = np.zeros([n_inter, n_inter])
    for i, alpha in enumerate(np.linspace(-1, 1, n_inter)):
        for j, beta in enumerate(np.linspace(-1, 1, n_inter)):
            loss_ls[i][j] = rand_2D_plot(model_0, alpha, beta)
    # print(loss_ls)
    
    plt.imsave("./result/Exp4_2D.png", loss_ls)
    


    