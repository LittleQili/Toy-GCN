import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as tutil

# from utils import accuracy
from smmodel import GCN
from smdata import load_data,accu

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,#加数据之后，需要训更多轮，这里15轮也不够，所以提升lr
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
BATCH_SIZE = 8
id,labels,adj_all,feature_all,data_all = load_data()
output_batch = torch.FloatTensor(BATCH_SIZE,1)
# print(output_batch.shape)
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
    output_batch = output_batch.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    loss_val = 0
    for i in range(0,adj_all.shape[0],BATCH_SIZE):
        optimizer.zero_grad()
        output = model(adj_all[i:i+BATCH_SIZE],feature_all[i:i+BATCH_SIZE])
        # print(output.shape)
        # print(labels[i:BATCH_SIZE])
        loss_train = mylossFunc(output,labels[i:i+BATCH_SIZE])
        loss_train.backward()
        optimizer.step()

        loss_val += loss_train.item()
        if i % 256 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
            # 'loss_train: {:.4f}'.format(loss_train.item()),
            # 'acc_val: ',(acc/256),
            'loss_val: {:.4f}'.format(loss_val*BATCH_SIZE/256),
            # 'acc_val: ',(acc/256),
            'time: {:.4f}s'.format(time.time() - t))
            loss_val = 0

    # loss_val = 0
    # acc = 0
    # j = 0
    # for i in range(adj_all.shape[0]):
    #     # for j in range(BATCH_SIZE):

    #     output = model(adj_all[i], feature_all[i])
    #     output_batch[j] = model(adj_all[i],feature_all[i])
    #     j += 1
    #     # print(j)
    #     if j == BATCH_SIZE:
    #         optimizer.zero_grad()
    #         loss_train = mylossFunc(output_batch,labels[i - j + 1:i+1])
    #         # if i==0:
    #         loss_train.backward(retain_graph=True)
    #         # else :
    #         #     loss_train.backward()
    #         optimizer.step()
    #         j = 0
    #         print('Epoch: {:04d}'.format(epoch+1),
    #         'loss_train: {:.4f}'.format(loss_train.item()),
    #         # 'acc_val: ',(acc/256),
    #         # 'loss_val: {:.4f}'.format(loss_val.item()/256),
    #         # 'acc_val: ',(acc/256),
    #         'time: {:.4f}s'.format(time.time() - t))
        # print(output)
        # print(labels[i])
        # print(type(output))<class 'torch.Tensor'>
        # print(output.shape) torch.Size([1])
        # print(labels[i].shape)
        # loss_train = mylossFunc(output, labels[i])
        # # acc_train = accuracy(output, labels[i])
        # if i % BATCH_SIZE == 0:
        #     loss_train.backward()
        #     optimizer.step()
        #     print('Epoch: {:04d}'.format(epoch+1),
        #     'loss_train: {:.4f}'.format(loss_train.item()),
        #     'acc_val: ',(acc/256),
        #     # 'loss_val: {:.4f}'.format(loss_val.item()/256),
        #     # 'acc_val: ',(acc/256),
        #     'time: {:.4f}s'.format(time.time() - t))
        # loss_val += mylossFunc(output, labels[i])
        # acc += accu(output,labels[i])
        # if i%256 == 0:
            
        # # acc_val = accuracy(output, labels[i])
        #     print('Epoch: {:04d}'.format(epoch+1),
        # #   'loss_train: {:.4f}'.format(loss_train.item()),
        # #   'acc_train: {:.4f}'.format(acc_train.item()),
        #   'loss_val: {:.4f}'.format(loss_val.item()/256),
        #   'acc_val: ',(acc/256),
        #   'time: {:.4f}s'.format(time.time() - t))
        #     loss_val = 0
        #     acc = 0  
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
    


# Train model
t_total = time.time()
model.train()
m_name = 'weight/yijiaGCN'
for epoch in range(args.epochs):
    train(epoch)
    if epoch % 1 == 0:
        torch.save(model,m_name+str(epoch)+'.pt')
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
torch.save(model,'weight/yijiaGCN.pt')