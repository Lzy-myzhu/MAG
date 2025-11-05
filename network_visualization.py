import torch
import torch.nn as nn
import graphviz
import os
import sys

# --- [ 您的配置和导入代码保持不变 ] ---

GRAPHVIZ_BIN_PATH = r'C:\Program Files\Graphviz\bin' # <--- 替换为您的实际路径！

if os.path.isdir(GRAPHVIZ_BIN_PATH):
    os.environ["PATH"] += os.pathsep + GRAPHVIZ_BIN_PATH
    print(f"Graphviz path explicitly set to: {GRAPHVIZ_BIN_PATH}")
else:
    print("Warning: Graphviz bin path not found at the specified location.")

# ⭐ 导入您的模型定义和配置 ⭐
try:
    from HeteroGAT_train_test import (
        HeteroGATModel, 
        node_types, 
        edge_types, 
        HIDDEN_DIM, 
        NUM_HEADS,
        DROPOUT_RATE,
        dataset, 
    )
except ImportError as e:
    # 假设使用默认值
    HIDDEN_DIM = 256
    NUM_HEADS = 4
    DROPOUT_RATE = 0.3
    class DummyDataset:
        num_classes = 349
    dataset = DummyDataset()
    node_types = ['paper', 'author', 'institution', 'field_of_study']
    edge_types = [
        ('paper', 'writes', 'author'), 
        ('paper', 'cites', 'paper'),
        ('author', 'affiliated_with', 'institution'),
        ('paper', 'has_topic', 'field_of_study')
    ]
    print(f"ERROR: 导入失败，使用默认配置。导入错误详情: {e}")
    
# 动态获取关系数量
NUM_RELATIONS = len(edge_types) 


# ----------------------------
# Helper function to create a HeteroConv Layer block (for symmetry)
# ----------------------------
def create_hetero_conv_block(dot, layer_num, input_node, output_node, hidden_dim, num_heads, num_relations, input_label):
    """
    创建单个 HeteroConv Layer 的详细块。
    """
    layer_name = f'cluster_conv{layer_num}'
    
    with dot.subgraph(name=layer_name) as g:
        g.attr(label=f'HeteroConv Layer {layer_num}', color='red', style='rounded')

        # 关系的输入
        g.node(f'input_rel_{layer_num}', label='Node Embeddings + Edge Indices Dict', shape='diamond', fillcolor='white')
        dot.edge(input_node, f'input_rel_{layer_num}', label=input_label)
        
        # 关系级 GAT Blocks
        gat_nodes = []
        for i in range(1, num_relations + 1):
            rel_label = f'GATConv for Relation {i}\n(Head-level Concatenation, Heads={num_heads})'
            node_name = f'gat_rel_{layer_num}_{i}'
            g.node(node_name, label=rel_label, fillcolor='lightyellow')
            g.edge(f'input_rel_{layer_num}', node_name, label=f'Relation {i} Edges')
            gat_nodes.append(node_name)

        # 关系聚合
        g.node(f'aggr_{layer_num}', 
                label='Relation-level Aggregation\n(Across Edge Types, Sum Pooling)', 
                fillcolor='lightcoral')
        
        # 从所有关系 GATConv 连接到聚合
        for node in gat_nodes:
            g.edge(node, f'aggr_{layer_num}', label='h_et')

    dot.node(output_node, label=f'h({layer_num}) for all node types\n(Dim: {hidden_dim})', shape='rect', fillcolor='white')
    dot.edge(f'aggr_{layer_num}', output_node)


# ----------------------------
# Model Architecture Visualization Function (Final Refined)
# ----------------------------
def plot_model_architecture_final_refined(
    model_name="Final HeteroGAT Architecture (OGBN-MAG)", 
    node_types=None, 
    edge_types=None, 
    hidden_dim=256, 
    num_heads=4, 
    dropout_p=0.3,
    num_classes=349,
    num_relations=4
):
    """
    绘制模型的高级架构图 (纵向布局，详细输入说明，对称 GNN 层)。
    """
    dot = graphviz.Digraph(comment=model_name, graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5'}) 
    dot.attr(compound='true')
    dot.attr('node', shape='box', style='filled', fontname='Helvetica', height='0.5')
    dot.attr('edge', arrowhead='vee', fontsize='10')
    
    # ------------------ 输入层 (修正特征描述) ------------------
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Features & Initial Embeddings (Mini-Batch)', color='blue', style='rounded')
        
        # 1. paper 节点的真实特征
        c.node('input_paper', 
               label='Paper Features (X_paper)\n(128-dim word2vec)', 
               fillcolor='palegreen')
        
        # 2. 其他节点的可学习嵌入
        other_nodes = [n for n in node_types if n != 'paper']
        for ntype in other_nodes:
            c.node(f'input_{ntype}', 
                   label=f'{ntype.capitalize()} Learnable Embeddings\n(nn.Embedding, No Original Features)', 
                   fillcolor='bisque')

        # 输入投影层
        input_proj_label = f'Input Feature Projection\n(Paper: Linear(In-Dim -> {hidden_dim}); Others: Identity/Lookup)'
        c.node('input_proj_layer', 
               label=input_proj_label, 
               shape='box', 
               fillcolor='lightgreen')

        # 连接到投影层
        c.edge('input_paper', 'input_proj_layer', label='True Features')
        for ntype in other_nodes:
            c.edge(f'input_{ntype}', 'input_proj_layer', label=f'Learnable Init')

    dot.node('h0', label=f'h(0) for all node types\n(Dim: {hidden_dim})', shape='rect', fillcolor='white')
    dot.edge('input_proj_layer', 'h0')
    dot.edge('h0', 'h0', label='Batch Data (NeighborLoader)', style='dashed', dir='none') 

    # ------------------ GNN Layer 1 (详细展示关系) ------------------
    create_hetero_conv_block(
        dot, 1, 'h0', 'h1_out', 
        hidden_dim, num_heads, num_relations, 
        input_label='h(0)'
    )
    
    # ------------------ Dropout Layer ------------------
    dot.node('dropout', 
             label=f'Dropout Layer\n(p={dropout_p})', 
             fillcolor='pink')
    dot.edge('h1_out', 'dropout')

    # ------------------ GNN Layer 2 (对称展示全部结构) ------------------
    create_hetero_conv_block(
        dot, 2, 'dropout', 'h2_out', 
        hidden_dim, num_heads, num_relations, 
        input_label=f'h\'(1) (Dropout Output)'
    )

    # ------------------ 输出层 ------------------
    
    # 最终 Paper 嵌入提取
    dot.node('output_paper_embedding', 
             label='Extract Final Paper Embeddings\n(h_paper_final)', 
             shape='box', 
             fillcolor='lightgray')
    dot.edge('h2_out', 'output_paper_embedding', label='Only take h_paper')
    
    # 分类器层
    dot.node('classifier', 
             label=f'Classifier (Linear Layer)\n({hidden_dim} -> {num_classes} classes)', 
             shape='box', 
             fillcolor='orange')
    dot.edge('output_paper_embedding', 'classifier', label='h_paper_final')

    # 最终输出
    dot.node('output_logits', 
             label=f'Logits for Target Paper Nodes\n(Batch Size: N_paper)', 
             shape='ellipse', 
             fillcolor='white')
    dot.edge('classifier', 'output_logits', label='Loss: Cross-Entropy (Target Nodes Only)')

    # 渲染并保存
    filename = model_name.replace(" ", "_").replace("/", "_").replace("-", "_")
    try:
        dot.render(filename, view=True, format='pdf', cleanup=True)
        print("\n" + "="*60)
        print(f"✅ 模型架构图已成功生成为 '{filename}.pdf'")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 警告：无法渲染 Graphviz 图。尝试手动保存 DOT 文件: '{filename}.dot'")
        dot.save(f'{filename}.dot')
        print(f"错误详情: {e}")
        print("="*60)


# ====================================================================
# ⭐ Main Execution Block
# ====================================================================
if __name__ == '__main__':
    
    # 确保 num_classes 是可用的
    num_classes = int(dataset.num_classes)
    
    # 调用新的绘图函数
    plot_model_architecture_final_refined(
        model_name="Final HeteroGAT Architecture (OGBN-MAG)", 
        node_types=node_types, 
        edge_types=edge_types, 
        hidden_dim=HIDDEN_DIM, 
        num_heads=NUM_HEADS, 
        dropout_p=DROPOUT_RATE,
        num_classes=num_classes,
        num_relations=NUM_RELATIONS
    )