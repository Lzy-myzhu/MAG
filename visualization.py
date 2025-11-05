import torch
import torch.nn.functional as F
import torch.nn as nn
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData 
import numpy as np 
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns 

# ----------------------------
# Config (必须与训练脚本保持一致)
# ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './dataset'
DATA_NAME = 'ogbn-mag'
HIDDEN_DIM = 256
NUM_HEADS = 4
DROP = 0.3 # 仅用于模型定义，可视化时不影响
NUM_NEIGHBORS = [10, 5] 
NUM_WORKERS = 0 # 可视化时通常将 workers 设为 0 以避免多进程问题
EVAL_BATCH_SIZE = 4096 # 较大的 batch size 以快速提取所有嵌入
BEST_MODEL_FILE = 'lr=0.01_batchsize=2048/mag_gat_best.pt'
# ----------------------------

# ----------------------------
# Helpers (必须与训练脚本保持一致)
# ----------------------------
def infer_metadata(data):
    if hasattr(data, 'num_nodes_dict') and hasattr(data, 'edge_index_dict'):
        node_types = list(data.num_nodes_dict.keys())
        edge_index_dict = dict(data.edge_index_dict)
        edge_types = list(edge_index_dict.keys())
        return node_types, edge_types, edge_index_dict
    raise RuntimeError("Cannot infer hetero metadata.")

def get_split_for_paper(split_idx):
    train = split_idx.get('train')
    valid = split_idx.get('valid')
    test = split_idx.get('test')
    if isinstance(train, dict):
        train = train.get('paper')
        valid = valid.get('paper')
        test  = test.get('paper')
    return train, valid, test

def index_to_tensor(idx):
    if isinstance(idx, np.ndarray):
        return torch.from_numpy(idx).to(torch.long)
    elif isinstance(idx, torch.Tensor):
        return idx.to(torch.long)
    else:
        return torch.tensor(idx, dtype=torch.long)

def add_reverse_relations(edge_index_dict):
    new = {}
    for (src, rel, dst), eidx in list(edge_index_dict.items()):
        rev_key = (dst, rel + '_rev', src)
        if rev_key not in edge_index_dict and rev_key not in new:
            rev_eidx = torch.stack([eidx[1], eidx[0]], dim=0).clone()
            new[rev_key] = rev_eidx
    edge_index_dict.update(new)
    return edge_index_dict

# ----------------------------
# 4. Model Definition (必须与训练脚本完全一致)
# ----------------------------
# 需要在数据加载后确定 emb_init_required 才能实例化模型，故先放在此处
# ... (代码省略了 HeteroGATModel 类定义，请确保它已存在于您的环境中或复制过来) ...
# 请将您训练脚本中的 'HeteroGATModel' 类的完整定义粘贴到这里，以确保一致性。
# ----------------------------------------------------------------------------------
class HeteroGATModel(nn.Module):
    def __init__(self, node_types, edge_types, num_nodes_dict, x_dict, hidden_dim=256, heads=4,
                 dropout=0.3, num_classes=349):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.num_classes = num_classes
        
        # 假设 emb_init_required 在外部已定义 (在 main 块中初始化)
        global emb_init_required 
        
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

        conv1_dict = {}
        conv2_dict = {}
        for et in self.edge_types:
            conv1_dict[et] = GATConv((-1, -1), hidden_dim // heads, heads=heads, add_self_loops=False)
            conv2_dict[et] = GATConv((-1, -1), hidden_dim // heads, heads=heads, add_self_loops=False)
        self.conv1 = HeteroConv(conv1_dict, aggr='sum')
        self.conv2 = HeteroConv(conv2_dict, aggr='sum')

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        x_in = {}
        for ntype, x in batch.x_dict.items():
            if ntype in self.embeddings:
                n_id = batch[ntype].n_id 
                x = self.embeddings[ntype](n_id) 
            
            x_in[ntype] = F.relu(self.input_proj[ntype](x))
        
        x1 = self.conv1(x_in, batch.edge_index_dict)
        x1 = {k: F.elu(v) for k, v in x1.items()}
        
        # 异构图卷积层 2 (没有 dropout，因为这是最终嵌入层)
        x2 = self.conv2(x1, batch.edge_index_dict)
        x2 = {k: F.elu(v) for k, v in x2.items()}
        
        # 返回最终嵌入
        return x2
# ----------------------------------------------------------------------------------


# ----------------------------
# 9. Visualization Functions
# ----------------------------
@torch.no_grad()
def get_embeddings_and_labels(model, data, index_tensor, index_len, num_neighbors, batch_size, device):
    """
    使用 NeighborLoader 提取模型在给定索引上的最后一层 GNN 嵌入和真实标签。
    """
    model.eval()
    
    loader = NeighborLoader(
        data, 
        num_neighbors=num_neighbors, 
        batch_size=batch_size, 
        input_nodes=('paper', index_tensor), 
        shuffle=False, 
        num_workers=NUM_WORKERS # 可视化时低 worker 更好
    )

    all_embeddings = []
    all_y_true = []
    
    for batch in loader:
        batch = batch.to(device)
        
        # 调用 forward，但只返回嵌入 (需要修改模型 forward 或直接调用到 conv2)
        # 这里使用修改后的 forward (它返回 x2 dict)
        x2 = model(batch)
        
        # 获取 Paper 节点的最终嵌入，并只取 Mini-Batch 核心节点
        embeddings = x2['paper'][:batch['paper'].batch_size]

        y_true = batch['paper'].y[:batch['paper'].batch_size]

        all_embeddings.append(embeddings.cpu())
        all_y_true.append(y_true.cpu())
    
    # 拼接并裁剪到实际长度
    embeddings = torch.cat(all_embeddings, dim=0).numpy()[:index_len]
    y_true = torch.cat(all_y_true, dim=0).squeeze().numpy()[:index_len]
    
    return embeddings, y_true

def visualize_embeddings(embeddings, labels, title="t-SNE Visualization of Node Embeddings"):
    """
    使用 t-SNE 对嵌入进行降维并在 2D 空间中可视化。
    """
    # 限制样本数量进行可视化，因为 t-SNE 对大规模数据计算量很大
    MAX_SAMPLES = 5000 
    
    if embeddings.shape[0] > MAX_SAMPLES:
        print(f"Sampling {MAX_SAMPLES} out of {embeddings.shape[0]} nodes for t-SNE...")
        # 随机采样 MAX_SAMPLES 个点
        indices = np.random.choice(embeddings.shape[0], MAX_SAMPLES, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    else:
        print(f"Using all {embeddings.shape[0]} nodes for t-SNE...")

    # ogbn-mag 类别数很多 (349)，我们只展示最常见的 N 个类别
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_n_labels = 20 
    
    # 确保类别数足够
    if len(unique_labels) < top_n_labels:
        top_n_labels = len(unique_labels)
        top_labels = unique_labels
    else:
        top_labels = unique_labels[np.argsort(counts)[-top_n_labels:]]

    # 过滤出需要展示的类别
    mask = np.isin(labels, top_labels)
    embeddings_filtered = embeddings[mask]
    labels_filtered = labels[mask]
    
    print(f"Applying t-SNE to {len(labels_filtered)} samples (Top {top_n_labels} classes)...")
    
    # 初始化 t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings_filtered)

    df = pd.DataFrame({
        'Dim 1': embeddings_2d[:, 0],
        'Dim 2': embeddings_2d[:, 1],
        'Label': labels_filtered.astype(str)
    })

    plt.figure(figsize=(12, 10))
    # 使用 seaborn 绘制散点图，用 Label 颜色区分
    sns.scatterplot(
        x='Dim 1', y='Dim 2',
        hue='Label',
        palette=sns.color_palette("hls", len(np.unique(labels_filtered))),
        data=df,
        legend="full",
        alpha=0.8,
        s=15 # 调整点大小
    )
    plt.title(f"{title} (Top {top_n_labels} Classes)")
    plt.legend(title='Venue ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='small')
    plt.tight_layout()
    #plt.show()
    plt.savefig('OGBN-MAG HeteroGAT t-SNE.png')

# ====================================================================
# ⭐ Main Execution Block
# ====================================================================
if __name__ == '__main__':
    if not os.path.isfile(BEST_MODEL_FILE):
        print(f"ERROR: Best model file '{BEST_MODEL_FILE}' not found.")
        print("Please run the training script first to generate the checkpoint.")
    else:
        print(f"Loading data and preparing graph structure...")
        
        # 1. Load dataset and extract components
        dataset = PygNodePropPredDataset(name=DATA_NAME, root=DATA_ROOT)
        split_idx = dataset.get_idx_split()
        data_old_format = dataset[0] 
        node_types, edge_types_raw, edge_index_dict_raw = infer_metadata(data_old_format)

        num_nodes_dict = data_old_format.num_nodes_dict
        x_dict_raw = data_old_format.x_dict
        y_dict_raw = data_old_format.y_dict

        train_idx_data, valid_idx_data, test_idx_data = get_split_for_paper(split_idx)
        # 我们只对测试集进行可视化
        test_idx = index_to_tensor(test_idx_data)
        test_len = test_idx.size(0)

        y_paper = y_dict_raw.get('paper').squeeze()

        # 2. Augment graph: add reverse edges
        edge_index_dict = {k: v.clone() for k, v in edge_index_dict_raw.items()}
        edge_index_dict = add_reverse_relations(edge_index_dict)
        edge_types = list(edge_index_dict.keys())

        # 3. Initialize features & Manual HeteroData Construction (OOM FIX logic)
        paper_feat = x_dict_raw.get('paper').float()
        feat_dim = paper_feat.size(1)
        x_dict = {}
        emb_init_required = {} 

        for ntype in node_types:
            n_nodes = int(num_nodes_dict[ntype])
            if ntype == 'paper':
                x_dict[ntype] = paper_feat
            else:
                x_dict[ntype] = torch.zeros((n_nodes, feat_dim), dtype=paper_feat.dtype) 
                emb_init_required[ntype] = n_nodes

        # 关键：手动创建 HeteroData 对象
        data = HeteroData()
        for ntype in node_types:
            data[ntype].num_nodes = num_nodes_dict[ntype]
            data[ntype].x = x_dict[ntype] 

        for et in edge_index_dict:
            data[et].edge_index = edge_index_dict[et]

        data['paper'].y = y_paper.unsqueeze(1) 
        data.num_nodes_dict = num_nodes_dict
        
        # 5. Load Model and State
        print(f"Loading best model from {BEST_MODEL_FILE}...")
        try:
            checkpoint = torch.load(BEST_MODEL_FILE, map_location=DEVICE)
            num_classes = int(dataset.num_classes)
            
            # 实例化模型
            model = HeteroGATModel(node_types, edge_types, num_nodes_dict, x_dict,
                                   hidden_dim=HIDDEN_DIM, heads=NUM_HEADS, dropout=DROP,
                                   num_classes=num_classes).to(DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            
            print(f"Model loaded successfully. Best Validation Accuracy: {best_val_acc:.4f}")
            
            # 6. Extract Embeddings and Visualize
            print("Starting Visualization...")
            
            final_embeddings, final_y_true = get_embeddings_and_labels(
                model, 
                data, 
                test_idx, 
                test_len, 
                NUM_NEIGHBORS, 
                EVAL_BATCH_SIZE, 
                DEVICE
            )

            visualize_embeddings(
                final_embeddings, 
                final_y_true, 
                title=f"OGBN-MAG HeteroGAT t-SNE (Best Val Acc: {best_val_acc:.4f})"
            )
            print("Visualization complete. Plot displayed.")

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model or visualize. Check if model class matches checkpoint structure.")
            print(f"Details: {e}")