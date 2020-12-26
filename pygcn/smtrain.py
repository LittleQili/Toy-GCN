
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as tutil

from utils import accuracy
from smmodel import GCN
from smdata import load_data

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
BATCH_SIZE = 16
id,labels,adj_all,feature_all,data_all = load_data()
tmplabels = labels.numpy()
labels = torch.FloatTensor(np.expand_dims(tmplabels,1))
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
# exit()
# Model and optimizer
model = GCN(norder=adj_all.shape[1],
            nfeat=feature_all.shape[2],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
mylossFunc = torch.nn.BCEWithLogitsLoss()
mydatainput = tutil.data.TensorDataset(data_all,labels)
mydataloader = tutil.data.DataLoader(
    dataset = mydatainput,
    batch_size = BATCH_SIZE
    )

if args.cuda:
    model.cuda()
    feature_all = feature_all.cuda()
    adj_all = adj_all.cuda()
    labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    for i in range(adj_all.shape[0]):
        model.train()
        optimizer.zero_grad()
        output = model(adj_all[i], feature_all[i])
        loss_train = mylossFunc(output, labels[i])
        # acc_train = accuracy(output, labels[i])
        loss_train.backward()
        optimizer.step()
        loss_val = mylossFunc(output, labels[i])
        # acc_val = accuracy(output, labels[i])
        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
        #   'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
        #   'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    # # 将模型转为训练模式，并将优化器梯度置零
    # # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    # # pytorch中每一轮batch需要设置optimizer.zero_grad
    # 
    # 
    # 

    # 

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    

    # output = torch.empty(BATCH_SIZE)
    # for i in range(BATCH_SIZE):
    #     output

    # for step ,(iinput,ilabel) in enumerate(mydataloader):
    #     tmpnpinput = iinput.numpy()
    #     adj = tmpnpinput[:,:,:132]
    #     fea = tmpnpinput[:,:,132:]
    #     adj = torch.FloatTensor(adj)
    #     fea = torch.FloatTensor(fea)

    #     output = []
    #     # output.append(model(adj[0],fea[0]).numpy())
    #     for i in range(adj.shape[0]):
    #         output.append(model(adj[i],fea[i]).numpy())
    #     output = np.array(output)
    #     output = torch.FloatTensor(output)
        

    # for i in range(adj_all.shape[0]):
    #     t = time.time()
    #     model.train()
    #     optimizer.zero_grad()

    #     output = model(feature_all[i], adj_all[i])
    #     # print('in func:',output.shape)
    #     loss_train = F.nll_loss(output, labels[i])
    #     acc_train = accuracy(output, labels[i])
    #     loss_train.backward()
    #     optimizer.step()
    #     if not args.fastmode:
    #         model.eval()
    #         output = model(feature_all[i], adj_all[i])
    #     loss_val = F.nll_loss(output, labels[i])
    #     acc_val = accuracy(output, labels[i])
    #     print('Epoch: {:04d}'.format(epoch+1),
    #           'loss_train: {:.4f}'.format(loss_train.item()),
    #           'acc_train: {:.4f}'.format(acc_train.item()),
    #           'loss_val: {:.4f}'.format(loss_val.item()),
    #           'acc_val: {:.4f}'.format(acc_val.item()),
    #           'time: {:.4f}s'.format(time.time() - t))
    


# def test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
