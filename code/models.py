import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,global_max_pool
from torch_geometric.nn import GCNConv
from sparse_softmax import Sparsemax
from torch_geometric.utils import softmax


#different x
class Edgelayer(nn.Module):
    def __init__(self, in_channels, sparse=True, negative_slop=0.2):
        super(Edgelayer, self).__init__()
        self.in_channels = in_channels  # related to num_nodes
        self.negative_slop = negative_slop
        self.sparse = sparse

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))

        nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()

    def forward(self, x, edge_index, edge_attr):

        row, col = edge_index  # inital row col

        weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
        weights = F.leaky_relu(weights, self.negative_slop)
        if self.sparse:
            new_edge_attr = self.sparse_attention(weights, row)
        else:
            new_edge_attr = softmax(weights, row, x.size(0))
        ind = torch.where(new_edge_attr != 0)[0]
        new_edge_index = edge_index[:, ind]
        new_edge_attr = new_edge_attr[ind]

        return x, new_edge_index, new_edge_attr


class EModel_block(torch.nn.Module):
    def __init__(self, args):
        super(EModel_block, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        self.edge = Edgelayer(self.num_features, sparse=True, negative_slop=0.2)

    def forward(self, x, edge_index, edge_attr, batch):
    
        x, new_edge_index, new_edge_attr = self.edge(x, edge_index, edge_attr)

        x = F.relu(self.conv1(x, new_edge_index, new_edge_attr))
        x = F.relu(self.conv2(x, new_edge_index, new_edge_attr))

        return x

# semantic attention
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=512):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)
       

class EModel(torch.nn.Module):
    def __init__(self, args):
        super(EModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.num_heads = args.num_heads

        self.graph_block1 = EModel_block(args)
        self.graph_block2 = EModel_block(args)
        self.graph_block3 = EModel_block(args)

        self.fc1 = nn.Linear(self.nhid * self.num_heads * 2, self.nhid * 2)
        self.fc2 = nn.Linear(self.nhid * 2, self.nhid)
        self.fc3 = nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        z1 = self.graph_block1(x, edge_index, edge_attr, batch)
        z2 = self.graph_block2(x, edge_index, edge_attr, batch)
        z3 = self.graph_block3(x, edge_index, edge_attr, batch)

        x1 = torch.cat([global_mean_pool(z1, batch), global_max_pool(z1, batch),
                        global_mean_pool(z2, batch), global_max_pool(z2, batch),
                        global_mean_pool(z3, batch), global_max_pool(z3, batch)], dim=1)

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.log_softmax(self.fc3(x1), dim=-1)

        return x1


class EModel_attention(torch.nn.Module):
    def __init__(self, args):
        super(EModel_attention, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.num_heads = args.num_heads

        self.graph_block1 = EModel_block(args)
        self.graph_block2 = EModel_block(args)
        self.graph_block3 = EModel_block(args)
        
        self.edge_attention = SemanticAttention(self.nhid*2)

        self.fc1 = nn.Linear(self.nhid*2, self.nhid * 2)
        self.fc2 = nn.Linear(self.nhid * 2, self.nhid)
        self.fc3 = nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        z1 = self.graph_block1(x, edge_index, edge_attr, batch)
        z2 = self.graph_block2(x, edge_index, edge_attr, batch)
        z3 = self.graph_block3(x, edge_index, edge_attr, batch)

        a1 = torch.cat([global_mean_pool(z1, batch), global_max_pool(z1, batch)], dim=1)
        a2 = torch.cat([global_mean_pool(z2, batch), global_max_pool(z2, batch)], dim=1)
        a3 = torch.cat([global_mean_pool(z3, batch), global_max_pool(z3, batch)], dim=1)
        
        x1 = torch.cat([a1.unsqueeze(1), a2.unsqueeze(1), a3.unsqueeze(1)], dim=1)
        
        x1 = self.edge_attention(x1)

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.log_softmax(self.fc3(x1), dim=-1)

        return x1
