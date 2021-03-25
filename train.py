import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as tutil

from myGCN import GCN
from dataproc import load_data,accu

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=7,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
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
BATCH_SIZE = 16
id,labels,adj_all,feature_all,data_all = load_data()

# Model and optimizer
model = GCN(norder=adj_all.shape[1],
            nfeat=feature_all.shape[2],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
mylossFunc = torch.nn.BCEWithLogitsLoss()

if args.cuda:
    model.cuda()
    feature_all = feature_all.cuda()
    adj_all = adj_all.cuda()
    labels = labels.cuda()

def train(epoch):
    t = time.time()
    loss_val = 0
    acc = 0
    for i in range(adj_all.shape[0]):
        optimizer.zero_grad()
        output = model(adj_all[i], feature_all[i])
        loss_train = mylossFunc(output, labels[i])
        loss_train.backward()
        optimizer.step()
        loss_val += mylossFunc(output, labels[i])
        acc += accu(output,labels[i])
        if i%256 == 0:
            
        # acc_val = accuracy(output, labels[i])
            print('Epoch: {:04d}'.format(epoch+1),
        #   'loss_train: {:.4f}'.format(loss_train.item()),
        #   'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()/256),
          'acc_val: ',(acc/256),
          'time: {:.4f}s'.format(time.time() - t))
            loss_val = 0
            acc = 0  

# Train model
t_total = time.time()
model.train()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
torch.save(model,'weight/yijiaGCN.pt')