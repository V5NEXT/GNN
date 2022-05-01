import pandas as pd
import matplotlib as plt
import numpy as np
from torch_geometric.datasets import CitationFull
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#from ge import DeepWalk

dataset = CitationFull(root='/tmp/Cora', name='Cora')
print("Length: " , len(dataset))
print("Num of classes: " , dataset.num_classes)
print("Num of node features: " , dataset.num_node_features)

data = dataset[0]
print("Undirected?: " , data.is_undirected())


train_dataset = dataset[:1200]
test_dataset = dataset[1200:]

perm = torch.randperm(len(dataset))
dataset = dataset[perm]


# train_mask denotes against which nodes to train
print("Train mask: " , data.train_mask.sum().item()) 

# val_mask denotes which nodes to use for validation, e.g., to perform early stopping
print("Val mask: " , data.val_mask.sum().item())

# test_mask denotes against which nodes to test
print("Test mask: " , data.test_mask.sum().item())

# GCN class
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

"""
The constructor defines two GCNConv layers which get called in the forward pass
 of our network. Note that the non-linearity is not integrated in the conv calls
 and hence needs to be applied afterwards (something which is consistent accross
 all operators in PyG). Here, we chose to use ReLU as our intermediate non-linearity
 and finally output a softmax distribution over the number of classes. Let’s train
 this model on the training nodes for 200 epochs:
"""    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
# evaluation

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

'''
# Visulazing 1
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

color_list = ["red", "orange", "green", "blue", "purple", "brown", "black"]

loader = DataLoader(dataset, batch_size=64, shuffle=True)
embs = []
colors = []
for batch in loader:
    emb, pred = model(batch)
    embs.append(emb)
    colors += [color_list[y] for y in batch.y]
embs = torch.cat(embs, dim=0)
print(emb)
xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
plt.scatter(xs, ys, color=colors)
'''

'''
# Visulizing 2

model.eval()
z = model.encode(x, data.train_pos_edge_index)
colors = [color_list[y] for y in labels]

xs, ys = zip(*TSNE().fit_transform(z.cpu().detach().numpy()))
plt.scatter(xs, ys, color=colors)
plt.show()
'''