import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData 
import copy
import numpy as np 
import os # 用于设置 num_workers 备选

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './dataset'
DATA_NAME = 'ogbn-mag'
HIDDEN_DIM = 256
NUM_HEADS = 4
DROP = 0.3
LR = 0.005
WEIGHT_DECAY = 1e-5
EPOCHS = 5
OUTPUT_PATH = 'lr=0.005_batchsize=1024'

# Mini-Batch Configuration 
BATCH_SIZE = 1024       
NUM_NEIGHBORS = [10, 5] 
NUM_WORKERS = 0 # 警告：如果运行失败，请将此值改为 0
# ----------------------------

# ----------------------------
# Helpers
# ----------------------------
def infer_metadata(data):
    """Infer node/edge types from old OGB Data format."""
    if hasattr(data, 'num_nodes_dict') and hasattr(data, 'edge_index_dict'):
        node_types = list(data.num_nodes_dict.keys())
        edge_index_dict = dict(data.edge_index_dict)
        edge_types = list(edge_index_dict.keys())
        return node_types, edge_types, edge_index_dict
    raise RuntimeError("Cannot infer hetero metadata.")

def get_split_for_paper(split_idx):
    """Extract paper node indices from OGB split dict."""
    train = split_idx.get('train')
    valid = split_idx.get('valid')
    test = split_idx.get('test')
    if isinstance(train, dict):
        train = train.get('paper')
        valid = valid.get('paper')
        test  = test.get('paper')
    return train, valid, test

def index_to_tensor(idx):
    """Handles conversion from numpy array or torch tensor to target torch.long tensor."""
    if isinstance(idx, np.ndarray):
        return torch.from_numpy(idx).to(torch.long)
    elif isinstance(idx, torch.Tensor):
        return idx.to(torch.long)
    else:
        return torch.tensor(idx, dtype=torch.long)

def add_reverse_relations(edge_index_dict):
    """Add reverse edges to the graph."""
    new = {}
    for (src, rel, dst), eidx in list(edge_index_dict.items()):
        rev_key = (dst, rel + '_rev', src)
        if rev_key not in edge_index_dict and rev_key not in new:
            rev_eidx = torch.stack([eidx[1], eidx[0]], dim=0).clone()
            new[rev_key] = rev_eidx
    edge_index_dict.update(new)
    return edge_index_dict

def save_checkpoint(epoch, model, optimizer, best_val_acc, filename='checkpoint.pt'):
    """保存当前 epoch 的模型和训练状态"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }
    torch.save(state, filename)
    print(f"--> Checkpoint saved to {filename} at epoch {epoch} with Val Acc {best_val_acc:.4f}")

def load_checkpoint(filename, model, optimizer):
    """加载检查点以继续训练"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'...")
        checkpoint = torch.load(filename, map_location=DEVICE)
        
        # 恢复状态
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        return start_epoch, best_val_acc
    else:
        print(f"No checkpoint found at '{filename}'. Starting training from scratch.")
        return 1, 0.0

# ----------------------------
# 1. Load dataset and extract components
# ----------------------------
dataset = PygNodePropPredDataset(name=DATA_NAME, root=DATA_ROOT)
split_idx = dataset.get_idx_split()
data_old_format = dataset[0] 
node_types, edge_types_raw, edge_index_dict_raw = infer_metadata(data_old_format)

num_nodes_dict = data_old_format.num_nodes_dict
x_dict_raw = data_old_format.x_dict
y_dict_raw = data_old_format.y_dict

train_idx_data, valid_idx_data, test_idx_data = get_split_for_paper(split_idx)
train_idx = index_to_tensor(train_idx_data)
valid_idx = index_to_tensor(valid_idx_data)
test_idx  = index_to_tensor(test_idx_data)

y_paper = y_dict_raw.get('paper').squeeze()

# ----------------------------
# 2. Augment graph: add reverse edges
# ----------------------------
edge_index_dict = {k: v.clone() for k, v in edge_index_dict_raw.items()}
edge_index_dict = add_reverse_relations(edge_index_dict)
edge_types = list(edge_index_dict.keys())
print("Node types:", node_types)
print("Edge types after adding reverse:", edge_types)

# ----------------------------
# 3. Initialize features & Manual HeteroData Construction (OOM FIX)
# ----------------------------
paper_feat = x_dict_raw.get('paper').float()
feat_dim = paper_feat.size(1)

x_dict = {}
# 这个字典用于告诉模型哪些节点类型需要 nn.Embedding
emb_init_required = {} 

for ntype in node_types:
    n_nodes = int(num_nodes_dict[ntype])
    
    if ntype == 'paper':
        # Paper 节点使用其原始特征
        x_dict[ntype] = paper_feat
    else:
        # **OOM FIX**: 非 Paper 节点统一使用可学习嵌入，跳过内存密集型聚合。
        x_dict[ntype] = torch.zeros((n_nodes, feat_dim), dtype=paper_feat.dtype) 
        emb_init_required[ntype] = n_nodes
        # print(f"[{ntype}] features skipped aggregation and will use learnable embeddings (size: {n_nodes}).")

# 关键：手动创建 HeteroData 对象
data = HeteroData()
for ntype in node_types:
    data[ntype].num_nodes = num_nodes_dict[ntype]
    # x 属性必须存在，即使是零填充
    data[ntype].x = x_dict[ntype] 

for et in edge_index_dict:
    data[et].edge_index = edge_index_dict[et]

data['paper'].y = y_paper.unsqueeze(1) 
data.num_nodes_dict = num_nodes_dict

print(f"✅ HeteroData prepared. Non-paper feature aggregation skipped.")


# ----------------------------
# 4. Define model (必须在 if __name__ == '__main__': 块外定义类)
# ----------------------------
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

        # 1. 节点特征映射/嵌入层
        self.input_proj = nn.ModuleDict()
        self.embeddings = nn.ModuleDict() 
        for ntype in self.node_types:
            if ntype in emb_init_required:
                # 对于需要嵌入的节点 (非paper)，使用 nn.Embedding
                n_nodes = emb_init_required[ntype]
                emb = nn.Embedding(n_nodes, hidden_dim)
                nn.init.xavier_uniform_(emb.weight)
                self.embeddings[ntype] = emb
                self.input_proj[ntype] = Linear(hidden_dim, hidden_dim)
            else:
                # 对于已有特征的 paper 节点
                in_dim = x_dict[ntype].size(1)
                self.input_proj[ntype] = Linear(in_dim, hidden_dim)

        # 2. HeteroGAT 层
        conv1_dict = {}
        conv2_dict = {}
        for et in self.edge_types:
            # GATConv(-1, -1) 用于异构图的灵活输入维度
            conv1_dict[et] = GATConv((-1, -1), hidden_dim // heads, heads=heads, add_self_loops=False)
            conv2_dict[et] = GATConv((-1, -1), hidden_dim // heads, heads=heads, add_self_loops=False)
        self.conv1 = HeteroConv(conv1_dict, aggr='sum')
        self.conv2 = HeteroConv(conv2_dict, aggr='sum')

        # 3. 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        x_in = {}
        for ntype, x in batch.x_dict.items():
            if ntype in self.embeddings:
                # 使用 Mini-Batch 提供的 n_id 查找全图 embedding
                n_id = batch[ntype].n_id 
                x = self.embeddings[ntype](n_id) 
            
            x_in[ntype] = F.relu(self.input_proj[ntype](x))
        
        # 异构图卷积层 1
        x1 = self.conv1(x_in, batch.edge_index_dict)
        x1 = {k: F.elu(v) for k, v in x1.items()}
        x1 = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x1.items()}
        
        # 异构图卷积层 2
        x2 = self.conv2(x1, batch.edge_index_dict)
        x2 = {k: F.elu(v) for k, v in x2.items()}
        
        # 仅对目标节点（'paper'）进行分类，且只取 Mini-Batch 核心节点
        out_paper = self.classifier(x2['paper'][:batch['paper'].batch_size])
        return out_paper

# ----------------------------
# 7. Train & Eval functions (在 if __name__ == '__main__': 块外定义)
# ----------------------------
def train_one_epoch():
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in train_loader:
        batch = batch.to(DEVICE) 
        optimizer.zero_grad()
        logits = model(batch) 
        y_true = batch['paper'].y[:batch['paper'].batch_size].squeeze(1)

        loss = F.cross_entropy(logits, y_true)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y_true.size(0)
        total_samples += y_true.size(0)
        
    return total_loss / total_samples

@torch.no_grad()
def evaluate(loader, index_len):
    model.eval()
    all_preds = []
    all_y_true = []
    
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        pred = logits.argmax(dim=-1, keepdim=True)
        y_true = batch['paper'].y[:batch['paper'].batch_size]

        all_preds.append(pred.cpu())
        all_y_true.append(y_true.cpu())
    
    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_y_true, dim=0).numpy()
    
    # 确保只使用有效的索引长度（Mini-Batch 评估可能产生额外长度）
    y_true = y_true[:index_len]
    y_pred = y_pred[:index_len]
    
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']


# ====================================================================
# ⭐ 8. Execution Block (必须在 if __name__ == '__main__': 保护块内)
# ====================================================================
if __name__ == '__main__':
    
    # ----------------------------
    # 4. Instantiate Model
    # ----------------------------
    num_classes = int(dataset.num_classes)
    model = HeteroGATModel(node_types, edge_types, num_nodes_dict, x_dict,
                           hidden_dim=HIDDEN_DIM, heads=NUM_HEADS, dropout=DROP,
                           num_classes=num_classes).to(DEVICE)
    
    # ----------------------------
    # 5. Prepare DataLoader 
    # ----------------------------
    print(f"Using {NUM_WORKERS} workers for DataLoader.")
    train_loader = NeighborLoader(
        data, 
        num_neighbors=NUM_NEIGHBORS, 
        batch_size=BATCH_SIZE, 
        input_nodes=('paper', train_idx), 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        drop_last=True
    )

    eval_batch_size = 2048 
    valid_loader = NeighborLoader(
        data, 
        num_neighbors=NUM_NEIGHBORS, 
        batch_size=eval_batch_size, 
        input_nodes=('paper', valid_idx),
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    test_loader = NeighborLoader(
        data, 
        num_neighbors=NUM_NEIGHBORS, 
        batch_size=eval_batch_size, 
        input_nodes=('paper', test_idx),
        shuffle=False, 
        num_workers=NUM_WORKERS
    )


    # ----------------------------
    # 6. Optimizer and evaluator
    # ----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    evaluator = Evaluator(name=DATA_NAME)

    # ----------------------------
    # 8. Training loop
    # ----------------------------
    CHECKPOINT_FILE = f'{OUTPUT_PATH}/mag_gat_latest.pt'
    BEST_MODEL_FILE = f'{OUTPUT_PATH}/mag_gat_best.pt'
    
    # 1. 尝试加载最新的训练进度 (恢复 start_epoch, best_val, model/optimizer 状态)
    start_epoch, best_val = load_checkpoint(CHECKPOINT_FILE, model, optimizer)
    
    # 2. 如果存在 'best' 模型文件，用它来初始化 best_state，并更新 best_val
    best_state = None
    best_epoch = start_epoch

    if os.path.isfile(BEST_MODEL_FILE):
        try:
            best_checkpoint_data = torch.load(BEST_MODEL_FILE, map_location=DEVICE)
            # 使用 best 模型的准确率来确保 best_val 不被降低
            best_val = max(best_val, best_checkpoint_data.get('best_val_acc', 0.0))
            # 将其状态字典保存为初始的 best_state
            best_state = best_checkpoint_data['model_state_dict']
            best_epoch = best_checkpoint_data.get('epoch', start_epoch)
            print(f"Initialized best_val from {BEST_MODEL_FILE}: {best_val:.4f} at epoch {best_epoch}")
        except Exception as e:
            print(f"Warning: Failed to load best checkpoint file {BEST_MODEL_FILE}. Error: {e}")

    # 3. 如果是首次训练 (没有 latest 和 best 文件)，则初始化 best_state 为当前模型的初始状态
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = start_epoch # 此时 start_epoch 必然是 1
        
    train_len = train_idx.size(0)
    valid_len = valid_idx.size(0)
    test_len = test_idx.size(0)

    print("Start Mini-Batch training on device:", DEVICE)
    print(f"Resuming from Epoch: {start_epoch}, Current Best Val: {best_val:.4f}")
    print(f"Num neighbors per layer: {NUM_NEIGHBORS}")

    # 4. 训练循环
    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        loss = train_one_epoch()
        
        # 验证和测试评估（可以在每次验证前加载最佳模型，但为了效率，通常直接在当前模型上评估）
        train_acc = evaluate(train_loader, train_len)
        valid_acc = evaluate(valid_loader, valid_len)
        test_acc = evaluate(test_loader, test_len)
        
        t1 = time.time()
        
        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Train/Val/Test {train_acc:.4f}/{valid_acc:.4f}/{test_acc:.4f} | Time {t1-t0:.1f}s")
        
        # 保存最新的检查点，方便断点续训
        save_checkpoint(epoch, model, optimizer, valid_acc, filename=CHECKPOINT_FILE)

        if valid_acc > best_val:
            best_val = valid_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            
            # 保存最佳模型
            save_checkpoint(epoch, model, optimizer, valid_acc, filename=BEST_MODEL_FILE)

    # final eval
    # 5. 最终评估：加载找到的最佳状态（无论是初始加载的，还是训练过程中发现的）
    if best_state is not None:
        model.load_state_dict(best_state)

        final_train = evaluate(train_loader, train_len)
        final_valid = evaluate(valid_loader, valid_len)
        final_test = evaluate(test_loader, test_len)

        print("\n" + "="*50)
        print(f"Best Epoch: {best_epoch}")
        print(f"Final results (best model): Train {final_train:.4f} | Valid {final_valid:.4f} | Test {final_test:.4f}")
        print("="*50)