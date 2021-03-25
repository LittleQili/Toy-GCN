import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, norder, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        # 输入输出
        self.norder = norder
        self.in_features = in_features
        self.out_features = out_features
        # 权重，可训练，即用parameter定义
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # bias，可训练，即用parameter定义
        if bias:
            self.bias = Parameter(torch.FloatTensor(norder,out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 初始化参数
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 为啥nm的是稀疏矩阵相乘？
    def forward(self, adj, fea):
        # 矩阵相乘，矩阵
        support = torch.mm(fea, self.weight)
        # 矩阵点积
        output = torch.mm(adj, support)
        # print('in-layers:',output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 转置之后再输进这个层
class MyFCN(Module):
    def __init__(self, dim1, dim2, bias=True):
        super(MyFCN, self).__init__()
        # 输入输出
        self.dim1 = dim1
        self.dim2 = dim2
        # 权重，可训练，即用parameter定义
        self.weight = Parameter(torch.FloatTensor(dim2, dim1))
        # bias，可训练，即用parameter定义
        if bias:
            self.bias = Parameter(torch.FloatTensor(1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 初始化参数
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, left):
        # 矩阵相乘
        output = torch.mm(left, self.weight)
        output = torch.sum(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.dim2) + ' -> ' \
               + str(self.dim1) + ')'


class GCN(nn.Module):
    def __init__(self,norder, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(norder, nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat, nclass)
        self.fcn = MyFCN(nclass,nhid)
        self.dropout = dropout

    def forward(self, adj, fea):
        x = F.relu(self.gc1(adj,fea))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.t()
        x = F.relu(self.gc2(x, fea))# ???
        x = x.t()
        x = self.fcn(x)
        return x