"""
@ref:
message forward: https://docs.dgl.ai/guide_cn/message.html
DGL forward functions: https://docs.dgl.ai/guide_cn/nn-forward.html#guide-cn-nn-forward
直接使用GAT的Layer: https://docs.dgl.ai/generated/dgl.nn.mxnet.conv.GATConv.html?highlight=gat#dgl.nn.mxnet.conv.GATConv
        from dgl.nn import GATConv
@note:
在DGL中，消息函数 接受一个参数 edges，这是一个 EdgeBatch 的实例， 在消息传递时，它被DGL在内部生成以表示一批边。 edges 有 src、 dst 和 data 共3个成员属性，
分别用于访问源节点、目标节点和边的特征。
聚合函数 接受一个参数 nodes，这是一个 NodeBatch 的实例， 在消息传递时，它被DGL在内部生成以表示一批节点。 nodes 的成员属性 mailbox 可以用来访问节点收到的消息。
一些最常见的聚合操作包括 sum、max、min 等。
更新函数 接受一个如上所述的参数 nodes。此函数对 聚合函数 的聚合结果进行操作， 通常在消息传递的最后一步将其与节点的特征相结合，并将输出作为节点的新特征。
DGL在命名空间 dgl.function 中实现了常用的消息函数和聚合函数作为 内置函数。 一般来说，DGL建议 尽可能 使用内置函数，因为它们经过了大量优化，并且可以自动处理维度广播。
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.nn.functional as fn
from functools import partial
from dgl.nn.pytorch.conv import GATv2Conv

class GATlayer(nn.Module):
    def __init__(self, config):
        super(GATlayer, self).__init__()
        feat_dim = config.hidden_size
        self.config = config
        self.graph_drop = 0 if not hasattr(self.config, "graph_drop") else self.config.graph_drop
        self.leaky_negative_slope = 0.98 if not hasattr(self.config, "leaky_negative_slope") \
            else self.config.leaky_negative_slope

        self.attn_drop = nn.Dropout(self.graph_drop)
        self.leaky_relu = nn.LeakyReLU(self.leaky_negative_slope)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # weight bases in equation (3)
        self.atten1 = nn.Linear(feat_dim + feat_dim, 1, bias=False)

    def forward(self, graph):
        def message_func(edges):
            return {'msg': edges.src['h_features'], 'a': edges.data['a']}

        def attention_message_func_node(edges):
            h = torch.cat([edges.src['h_features'][:, 0], edges.dst['h_features'][:, 0]], dim=1)
            a = self.atten1(h)
            a = self.attn_drop(a)
            return {'a': self.leaky_relu(a)}

        def apply_func(nodes):
            alpha = F.softmax(nodes.mailbox['a'], dim=1)
            alpha = alpha.unsqueeze(-1).repeat((1, 1, nodes.mailbox['msg'].shape[2], nodes.mailbox['msg'].shape[3]))
            h_features = torch.sum(alpha * nodes.mailbox['msg'], dim=1)
            return {'h_features': h_features}

        graph.apply_edges(attention_message_func_node)
        graph.update_all(message_func, apply_func)
        return graph