from ogb.nodeproppred import PygNodePropPredDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_dim=64, out_dim=349):  # ogbn-mag有349类
        super().__init__()
        self.hidden_dim = hidden_dim

        # 第一层
        self.conv1 = HeteroConv({
            edge_type: GATConv((-1, -1), hidden_dim, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')

        # 第二层
        self.conv2 = HeteroConv({
            edge_type: GATConv((-1, -1), out_dim, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # 第一层传播
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        # 第二层传播
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


def get_graph(name='ogbn-mag'):
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0]
    print(graph)
    return graph

get_graph()