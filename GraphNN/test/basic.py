import torch
from torch_geometric.data import Data

# 边的连接信息
# 注意，无向图的边要定义两次
edge_index = torch.tensor(
    [
        # 这里表示节点0和1有连接，因为是无向图
        # 那么1和0也有连接
        # 上下对应着看
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ],
    # 指定数据类型
    dtype=torch.long
)
# 节点的属性信息
x = torch.tensor(
    [
        # 三个节点
        # 每个节点的属性向量维度为1
        [-1],
        [0],
        [1],
    ]
)
# 实例化为一个图结构的数据
data = Data(x=x, edge_index=edge_index)
# 查看图数据
print(data)
# 图数据中包含什么信息
print(data.keys)
# 查看节点的属性信息
print(data['x'])
# 节点数
print(data.num_nodes)
# 边数
print(data.num_edges)
# 节点属性向量的维度
print(data.num_node_features)
# 图中是否有孤立节点
print(data.has_isolated_nodes())
# 图中是否有环
print(data.has_self_loops())
# 是否是有向图
print(data.is_directed())

# curl &apos;http://10.9.1.3&apos; --data "DDDDD= 20215227045&upass=dyh19980220@&0MKKey="
