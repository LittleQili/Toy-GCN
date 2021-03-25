import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as tutil

from myGCN import GCN
from dataproc import load_test_data

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

id,adj_smiles,feature_smiles,allinput = load_test_data()

model = torch.load('weight/yijiaGCN.pt')

if args.cuda:
    # model.cuda()
    feature_smiles = feature_smiles.cuda()
    adj_smiles = adj_smiles.cuda()

finalact = torch.nn.Sigmoid()

f = open('output_518030910146.txt','w')
f.write('Chemical,Label\n')
for i in range(adj_smiles.shape[0]):
    output = model(adj_smiles[i],feature_smiles[i])
    output = finalact(output)[0].item()
    f.write(id[i] + ',%f\n' % output)
f.close()