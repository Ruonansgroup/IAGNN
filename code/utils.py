import numpy as np
from torch_geometric.data import Data
import torch
import scipy.io as io
from torch_geometric.utils import dense_to_sparse


# dimension of x_set [feature_size, num_nodes, num_sample]
# raw data label start from index 1
def data_process(raw_path, num_nodes):
    data = io.loadmat(raw_path)
    if raw_path[-17:-8] == 'train_set':
        x_set = data['x_train']
        y_set = data['y_train']
    elif raw_path[-17:-8] == 'valid_set':
        x_set = data['x_valid']
        y_set = data['y_valid']
    elif raw_path[-16:-8] == 'test_set':
        x_set = data['x_test']
        y_set = data['y_test']

    y_set = np.squeeze(np.array(y_set))
    y_set = y_set - 1
    labels = torch.tensor(y_set, dtype=torch.long)
    init_adj = torch.ones((num_nodes, num_nodes), dtype=torch.float) - torch.eye(num_nodes, dtype=torch.float)
    edge_index, _ = dense_to_sparse(init_adj)
    graph_list = []
    for k in range(labels.size(0)):
        x = x_set[:, :, k]
        #x = x[:, 0:-1]
        x = torch.tensor(x, dtype=torch.float)
        x = torch.transpose(x, 1, 0)
        g = Data(x=x, edge_index=edge_index, y=labels[k])
        graph_list.append(g)

    return graph_list
    




