import pysmiles as psm
import networkx as nx
import numpy as np
import os
import torch

'''
data
化学式数目：8169
化学式最长长度(element个数)：132
特征矩阵维度：4

特征：
element: need a static dict to stand for each element
aromatic: 直接用
hcount: 直接用
charge: 非常稀疏，偶尔有个-1之类的，可以考虑不用
isotope& class 两个装死

特征扩展：主要目的是补齐空缺位置,这个是必须的，不然会报错
目前：feature 0扩展，维度为(4,132)
adj 先0扩展再进行计算，维度为(132,132)

方案1：adj矩阵扩展为0，feature矩阵也扩展为0
方案2：读一读adj来源图，考虑一下扩展成-1？之类的，或者对角线-1？
      feature矩阵考虑扩展成-1？但是需要注意charge的-1,扩展可能不太一样

smile CC=CCC#N
element:  [(0, 'C'), (1, 'C'), (2, 'C'), (3, 'C'), (4, 'C'), (5, 'N')]
aromatic:   [(0, False), (1, False), (2, False), (3, False), (4, False), (5, False)]
isotope:   [(0, None), (1, None), (2, None), (3, None), (4, None), (5, None)]
hcount:   [(0, 3), (1, 1), (2, 1), (3, 2), (4, 0), (5, 0)]
charge:   [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
class:   [(0, None), (1, None), (2, None), (3, None), (4, None), (5, None)]
adj_h:
 [[0.5        0.5        0.         0.         0.         0.        ]
 [0.25       0.25       0.5        0.         0.         0.        ]
 [0.         0.5        0.25       0.25       0.         0.        ]
 [0.         0.         0.33333333 0.33333333 0.33333333 0.        ]
 [0.         0.         0.         0.2        0.2        0.6       ]
 [0.         0.         0.         0.         0.75       0.25      ]]
'''

'''
Question:
the expanded feature array is too sparce
'''

dict_element = {'C': 1, 'N': 2, 'O': 3, 'Br': 4, 'Cl': 5, 'Na': 6, 'S': 7, 'P': 8, 'Ca': 9, 'F': 10, 'B': 11, 'H': 12,
               'As': 13, 'Al': 14, 'I': 15, 'Si': 16, 'K': 17, 'Cr': 18, 'Zn': 19, 'Se': 20, 'Zr': 21, 'Fe': 22,
               'Sn': 23, 'Nd': 24, 'Cu': 25, 'Au': 26, 'Pb': 27, 'Tl': 28, 'Sb': 29, 'Cd': 30, 'Pd': 31, 'Ti': 32,
               'Pt': 33, 'In': 34, 'Ba': 35, 'Ag': 36, 'Dy': 37, 'Hg': 38, 'Li': 39, 'Yb': 40, 'Mn': 41, 'Mg': 42,
               'Co': 43, 'Ni': 44, 'Be': 45, 'Ge': 46, 'Bi': 47, 'V': 48, 'Sr': 49, 'Mo': 50, 'Ru': 51, 'Eu': 52,
               'Sc': 53}
# from test
Maximum_length_smile = 132
# Maximum_length = 0
tmpnum_smiles = 32

def fread_smiles(path):
    with open(path, 'r') as f:
        dts = f.read().split('\n')[1:]

    names_to_smiles = {}
    for dt in dts:
        if dt == '':
            continue
        data = dt.split(',')
        names_to_smiles[data[0]] = data[1]

    return names_to_smiles

def fread_labels(path):
    file = open(path, 'r').read()
    data_list = file.split('\n')[1:]
    return_dict = {}
    for data in data_list:
        if data:
            list_d = data.split(',')
            return_dict[list_d[0]] = int(list_d[1])

    return return_dict

def proc_one_smile(smile_nx):
    element = smile_nx.nodes(data='element')
    isotope = smile_nx.nodes(data='isotope')
    aromatic = smile_nx.nodes(data='aromatic')
    hcount = smile_nx.nodes(data='hcount')
    charge = smile_nx.nodes(data='charge')
    _class = smile_nx.nodes(data='class')
    
    feature = []
    fet_element = []
    fet_aromatic = []
    fet_hcount = []
    fet_charge = []
    for i in element:
        fet_element.append(dict_element[i[1]])
    for i in aromatic:
        if i[1]:
            fet_aromatic.append(1)
        else:
            fet_aromatic.append(0)
    for i in hcount:
        fet_hcount.append(i[1])
    for i in charge:
        fet_charge.append(i[1])
    feature.append(fet_element)
    feature.append(fet_aromatic)
    feature.append(fet_hcount)
    feature.append(fet_charge)
    feature = np.array(feature)

    # print(feature.shape)
    # print(feature)

    global Maximum_length_smile
    tmp = int(Maximum_length_smile - feature.shape[1])
    expand_feature = np.zeros((4,tmp))
    feature = np.concatenate((feature,expand_feature),axis = 1)
    feature = feature.transpose()

    # print(feature.shape)
    # print(feature)
    # print(expand_feature.shape)
    # print(expand_feature)
    # print()

    # global Maximum_length
    # if feature.shape[1] > Maximum_length :
    #     Maximum_length = feature.shape[1]

    # print(feature.shape)
    # print(feature)                  
    # print('element:',type(element),element)
    # print('aromatic:',type(aromatic),aromatic)
    # print('isotope:',type(isotope),isotope)
    # print('hcount:',type(hcount),hcount)
    # print('charge:',type(charge),charge)
    # print('class:',type(_class),_class)

    adj = nx.to_numpy_matrix(smile_nx,weight='order')
    # print(adj.shape)
    
    # print(adj.shape)
    # print(adj)
    # print('adj:',type(adj),'\n',adj)

    adj = adj+np.eye(adj.shape[0])
    # print('adj:',type(adj),'\n',adj)

    sum_ = np.array(np.sum(adj, axis=0)).flatten()
    # print('sum_:',type(sum_),'\n',sum_)
    row_sum = np.power(np.diag(sum_), -1)
    # print('row_sum:',type(row_sum),'\n',row_sum)
    row_sum[np.isinf(row_sum)] = 0
    # print('row_sum:',type(row_sum),'\n',row_sum)
    adj = np.dot(row_sum, adj)
    # print('adj_h:',type(adj_h),'\n',adj_h)

    expand_feature = np.zeros((adj.shape[0],Maximum_length_smile-adj.shape[1]))
    adj = np.concatenate((adj,expand_feature),axis = 1)
    expand_feature = np.zeros((Maximum_length_smile-adj.shape[0],adj.shape[1]))
    adj = np.concatenate((adj,expand_feature),axis = 0)

    return adj, feature
    

def load_data():
    ffolder_train = "../data/train/"

    fname_smiles = "names_smiles.txt"
    fname_onehot = "names_onehots.npy"
    fname_label = "names_labels.txt"

    dict_id_smiles = fread_smiles(ffolder_train+fname_smiles)
    # print(dict_id_smiles)
    smiles = []# smile string, for debug
    id = []# chemical id
    mol_smiles = []# debug
    adj_smiles = []#adjacent
    feature_smiles = []#feature
    allinput = []
    tmpi = 0
    for k in dict_id_smiles:
        # print('smile',dict_id_smiles[k])
        smiles.append(dict_id_smiles[k])
        id.append(k)
        mol_smiles.append(psm.read_smiles(dict_id_smiles[k]))
        tmpadj,tmpfet = proc_one_smile(psm.read_smiles(dict_id_smiles[k]))
        adj_smiles.append(tmpadj)
        feature_smiles.append(tmpfet)
        cat = np.concatenate((tmpadj,tmpfet),axis=1)
        allinput.append(cat)
        tmpi+=1
        if tmpi >= tmpnum_smiles:
            break
        # break
    
    dict_id_label = fread_labels(ffolder_train+fname_label)
    label_list = []
    tmpi = 0
    for k in dict_id_label:
        label_list.append(dict_id_label[k])
        tmpi+=1
        if tmpi >= tmpnum_smiles:
            break
    # print('label_list:',label_list)

    # adj_smiles = np.array(adj_smiles)
    # adj_smiles = adj_smiles.astype(float)
    # _adj_smiles = torch.FloatTensor(adj_smiles)
    adj_smiles = torch.FloatTensor(np.array(adj_smiles))
    feature_smiles = torch.FloatTensor(np.array(feature_smiles))
    allinput = torch.FloatTensor(np.array(allinput))
    labels = torch.LongTensor(np.array(label_list))
    print(adj_smiles.shape)
    print(feature_smiles.shape)
    print(labels.shape)
    print(allinput.shape)
    '''
    torch.Size([8169, 132, 132])
    torch.Size([8169, 4, 132])
    torch.Size([8169])
    torch.Size([32, 136, 132])
    '''
    print(feature_smiles.shape[2])
    print(labels.max().item() + 1)
    return id,labels,adj_smiles,feature_smiles,allinput

load_data()
# print(Maximum_length)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)