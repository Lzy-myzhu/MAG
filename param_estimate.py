#此代码跑不通


from torchsummary import summary
from torch_geometric.nn import HeteroConv, GATConv, Linear
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroGATModel(nn.Module):
    def __init__(self, node_types, edge_types, num_nodes_dict, x_dict,
                 hidden_dim=256, heads=4, dropout=0.3, num_classes=349):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.num_classes = num_classes

        emb_init_required = {ntype: num_nodes_dict[ntype]
                             for ntype in node_types if ntype not in x_dict}

        self.input_proj = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        for ntype in self.node_types:
            if ntype in emb_init_required:
                n_nodes = emb_init_required[ntype]
                emb = nn.Embedding(n_nodes, hidden_dim)
                nn.init.xavier_uniform_(emb.weight)
                self.embeddings[ntype] = emb
                self.input_proj[ntype] = Linear(hidden_dim, hidden_dim)
            else:
                in_dim = x_dict[ntype].size(1)
                self.input_proj[ntype] = Linear(in_dim, hidden_dim)

        conv1_dict, conv2_dict = {}, {}
        for et in self.edge_types:
            conv1_dict[et] = GATConv((-1, -1), hidden_dim // heads, heads=heads, add_self_loops=False)
            conv2_dict[et] = GATConv((-1, -1), hidden_dim // heads, heads=heads, add_self_loops=False)
        self.conv1 = HeteroConv(conv1_dict, aggr='sum')
        self.conv2 = HeteroConv(conv2_dict, aggr='sum')
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        # 不执行实际前向传播
        raise NotImplementedError("For summary, we don't need forward pass.")


# ==========================================================
# 模型构造
# ==========================================================
node_types = ['paper', 'author', 'institution', 'field_of_study']
edge_types = [
    ('author', 'writes', 'paper'),
    ('paper', 'cites', 'paper'),
    ('paper', 'has_topic', 'field_of_study'),
    ('author', 'affiliated_with', 'institution'),
    ('institution', 'employs', 'author'),
    ('field_of_study', 'rev_topic', 'paper'),
    ('paper', 'written_by', 'author'),
    ('paper', 'referenced_by', 'paper')
]
num_nodes_dict = {
    'paper': 736389,
    'author': 1134649,
    'institution': 8740,
    'field_of_study': 59965
}
x_dict = {'paper': torch.zeros((10, 128))}

model = HeteroGATModel(node_types, edge_types, num_nodes_dict, x_dict)

# ==========================================================
# torchsummary 输出
# ==========================================================
summary(model, input_size=(1,), device='cpu', verbose=1)
