import jax
import jax.numpy as jnp
import jraph

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import numpy as np
import copy
import os

# load torch_geometic dataset
def load_graph_from_torch(name, path, homophily=None, degree=None):
    import torch_geometric.transforms as T

    if name in ["Cora", "CiteSeer", "PubMed"]:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ["chameleon", "squirrel"]:
        from torch_geometric.datasets import WikipediaNetwork
        dataset = WikipediaNetwork(path, name, transform = T.NormalizeFeatures())
    elif name in ["Wisconsin", "Cornell", "Texas"]:
        from torch_geometric.datasets import  WebKB
        dataset = WebKB(path, name, transform = T.NormalizeFeatures())
    elif name in ['Actor']:
        from torch_geometric.datasets import Actor
        dataset = Actor(path, transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError("Not Implemented Dataset!")
    
    return dataset

# Get graph tuples from torch_geometric dataset
def get_single_graph_tuples(dataset):
    # dataset should consist of single graph ()
    assert len(dataset) == 1
    single_graph = dataset[0] # keys: x, edge_index, y, train_mask, val_mask, test_mask
    
    forward_edge = single_graph.edge_index.t()[single_graph.edge_index[0] < single_graph.edge_index[1]].t()
    backward_edge = torch.stack((forward_edge[1],forward_edge[0]),dim=0)
    edges = torch.cat((forward_edge,backward_edge),dim=1)
    
    assert (forward_edge[0] - backward_edge[1]).sum() == 0
    assert (forward_edge[1] - backward_edge[0]).sum() == 0

    graphs = jraph.GraphsTuple(
            nodes=jnp.asarray(single_graph.x.numpy()),
            edges=None, # this particular data doesn't have edge features, hence we set to None
            n_node=jnp.asarray([single_graph.x.size(0)]),
            n_edge=jnp.asarray([edges.size(1)]),
            senders=jnp.asarray(edges[0].numpy()), 
            receivers=jnp.asarray(edges[1].numpy()), 
            globals=None)
    
    n_cls = single_graph.y.max().item() + 1
    labels = jnp.array(torch.nn.functional.one_hot(single_graph.y, n_cls).numpy())

    return graphs, labels

# Get train/val/test masks from torch_geometric dataset
def get_split_mask(dataset, name, path, imb_ratio=1, split_num=None):
    # dataset should consist of single graph ()
    assert len(dataset) == 1
    single_graph = dataset[0] # keys: x, edge_index, y, train_mask, val_mask, test_mask
    
    if name in ["Cora", "CiteSeer", "PubMed"]:
        train_mask = jnp.asarray(single_graph.train_mask.float().numpy())
        val_mask = jnp.asarray(single_graph.val_mask.float().numpy())
        test_mask = jnp.asarray(single_graph.test_mask.float().numpy())
    elif name in ["chameleon", "squirrel", "Wisconsin", "Cornell", "Texas", 'Actor']:
        train_mask = jnp.asarray(single_graph.train_mask[:,split_num%10].float().numpy())
        val_mask = jnp.asarray(single_graph.val_mask[:,split_num%10].float().numpy())
        test_mask = jnp.asarray(single_graph.test_mask[:,split_num%10].float().numpy())
    else:
        raise NotImplementedError("Not Implemented Dataset!")
    
    return train_mask, val_mask, test_mask


# Identify edges which affect the representations of train nodes
def get_effect_edge_idx(graph, _train_mask):
    
    half_n_edge = graph.n_edge[0] // 2
    half_senders = np.array(graph.senders[:half_n_edge])
    half_receivers = np.array(graph.receivers[:half_n_edge])
    train_mask = np.array(_train_mask).astype(bool)
    
    # 1-hop edges from train nodes
    train_senders = train_mask[half_senders].astype(bool)
    train_receivers = train_mask[half_receivers].astype(bool)
    train_edge_mask = train_senders | train_receivers
    
    # 2-hop edges from train nodes
    train_node_idx = np.concatenate(
        (half_senders[train_edge_mask.astype(bool)],half_receivers[train_edge_mask.astype(bool)]),
        axis=0)
    train_node_idx = np.unique(train_node_idx, axis=0)
    train_mask_2hop = copy.deepcopy(train_mask).astype(bool)
    train_mask_2hop[train_node_idx] = True
    train_2hop_senders = train_mask_2hop[half_senders].astype(bool)
    train_2hop_receivers = train_mask_2hop[half_receivers].astype(bool)
    train_2hop_edge_mask = train_2hop_senders | train_2hop_receivers
    train_2hop_edge_idx = np.arange(half_n_edge)[train_2hop_edge_mask.astype(bool)]
    
    return train_2hop_edge_idx