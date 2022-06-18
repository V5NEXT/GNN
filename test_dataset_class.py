import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from tqdm import tqdm


class Frankenstein(InMemoryDataset):
    
    # Base url to download the files
    url = 'http://nrvis.com/download/data/labeled/FRANKENSTEIN.zip'
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(Frankenstein, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['FRANKENSTEIN.edges', 'FRANKENSTEIN.graph_idx',
                'FRANKENSTEIN.graph_labels', 'FRANKENSTEIN.node_attrs']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        # The zip file is removed
        os.unlink(path)


    def process(self):
        # Read the files' content as Pandas DataFrame. Nodes and graphs ids
        # are based on the file row-index, we adjust the DataFrames indices
        # by starting from 1 instead of 0.
        
        path = os.path.join(self.raw_dir, 'FRANKENSTEIN.node_attrs')
        node_attrs = pd.read_csv(path, sep=',', header=None)
        node_attrs.index += 1
        
        path = os.path.join(self.raw_dir, 'FRANKENSTEIN.edges')
        edge_index = pd.read_csv(path, sep=',', names=['source', 'target'])
        edge_index.index += 1
        
        path = os.path.join(self.raw_dir, 'FRANKENSTEIN.graph_idx')
        graph_idx = pd.read_csv(path, sep=',', names=['idx'])
        graph_idx.index += 1
        
        path = os.path.join(self.raw_dir, 'FRANKENSTEIN.graph_labels')
        graph_labels = pd.read_csv(path, sep=',', names=['label'])
        graph_labels.index += 1
        
        
        # In the loop we extract the nodes' embeddings, edges connectivity for 
        # and label for a graph, process the information and put it in a Data
        # object, then we add the object to a list
        data_list = []
        ids_list = graph_idx['idx'].unique()
        for g_idx in tqdm(ids_list):
            node_ids = graph_idx.loc[graph_idx['idx']==g_idx].index
            
            # Node features
            attributes = node_attrs.loc[node_ids, :]
            
            # Edges info
            edges = edge_index.loc[edge_index['source'].isin(node_ids)]
            edges_ids = edges.index
            
            # Graph label
            label = graph_labels.loc[g_idx]
            
            # Normalize the edges indices
            edge_idx = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            map_dict = {v.item():i for i,v in enumerate(torch.unique(edge_idx))}
            map_edge = torch.zeros_like(edge_idx)
            for k,v in map_dict.items():
                map_edge[edge_idx==k] = v
            
            # Convert the DataFrames into tensors 
            attrs = torch.tensor(attributes.to_numpy(), dtype=torch.float)
            pad = torch.zeros((attrs.shape[0], 4), dtype=torch.float)
            x = torch.cat((attrs, pad), dim=-1)

            edge_idx = map_edge.long()

            np_lab = label.to_numpy()
            y = torch.tensor(np_lab if np_lab[0] == 1 else [0], dtype=torch.long)
            
            graph = Data(x=x, edge_index=edge_idx,  y=y)
            
            data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  