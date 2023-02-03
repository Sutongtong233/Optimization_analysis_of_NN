import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv, GCNConv, SAGEConv

class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.use_bias = bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)
        output = torch.mm(adj, support)
        if self.use_bias:
            return output + self.bias
        else:
            return output


class SGC(torch.torch.nn.Module):
    def __init__(self, input_dim=1433, hidden_dim=16, output_dim=7):
        super(SGC, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)
        pass
    
    def forward(self, X, adj):
        X = self.gcn1(X, adj)
        X = F.relu(X)
        X = self.gcn2(X, adj)
        return X

       

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GRAPH(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, input_dim = 1433, hidden_dim = 16, output_dim = 7):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


