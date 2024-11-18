from typing import Union, List, Tuple

import torch
import scipy
import numpy as np
import networkx as nx
from networkx.algorithms import tree
import torch_geometric
import torch_geometric.typing
from torch.nn import MSELoss
from torch_geometric.utils import negative_sampling, to_undirected, to_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from sklearn import preprocessing
from torch_scatter import scatter_add
from torch_sparse import coalesce

import scipy.io as sio
import scipy.stats as st

from typing import Optional, Callable
import torcheval.metrics.functional
import torch_geometric.utils
import torch_geometric.utils.num_nodes
from torch_geometric_signed_directed.data import SignedData
from torch_geometric_signed_directed.data.signed import SDGNN_real_data
from torch_geometric_signed_directed.data import SSSNET_real_data


import flipping
import antiparallel

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def acc(pred, label, weighted=False):
    if weighted == True:
        acc = torcheval.metrics.functional.r2_score(pred.squeeze(), label.squeeze()).detach().item()
    else:
        #print(pred.shape, label.shape)       
        correct = pred.eq(label).sum().item()
        acc = correct / len(pred)
    return acc

def in_out_degree(edge_index, size, weight=None):
    if weight is None:
        A = coo_matrix((np.ones(len(edge_index)), (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()
    else:
        A = coo_matrix((weight, (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()

    out_degree = np.sum(np.abs(A), axis = 0).T
    in_degree = np.sum(np.abs(A), axis = 1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree

def undirected_label2directed_label(adj: scipy.sparse.csr_matrix, edge_pairs: List[Tuple],
                                    task: str, directed_graph: bool = True, signed_directed: bool = False, keep_negatives: bool = False) -> Union[List, List]:
    r"""Generate edge labels based on the task.
    Arg types:
        * **adj** (scipy.sparse.csr_matrix) - Scipy sparse undirected adjacency matrix. 
        * **edge_pairs** (List[Tuple]) - The edge list for the link dataset querying. Each element 
            in the list is an edge tuple.
        * **edge_weight** (List[Tuple]) - The edge weights list for sign graphs.
        * **task** (str): three_class_digraph (three-class link prediction); direction (direction prediction); existence (existence prediction); sign (sign prediction); 
            four_class_signed_digraph (directed sign prediction); five_class_signed_digraph (directed sign and existence prediction) 
    Return types:
        * **new_edge_pairs** (List) - A list of edges.
        * **labels** (List) - The labels for new_edge_pairs. 
            * If task == "existence": 0 (the directed edge exists in the graph), 1 (the edge doesn't exist).
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "direction": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists).
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "three_class_digraph": 0 (the directed edge exists in the graph), 
                1 (the edge of the reversed direction exists), 2 (the edge doesn't exist in both directions). 
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "four_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                3 (the edge of the reversed direction exists). 
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "five_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                3 (the edge of the reversed direction exists), 4 (the edge doesn't exist in both directions). 
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "sign": 0 (negative edge), 1 (positive edge). 
        * **label_weight** (List) - The weight list of the query edges. The weight is zero if the directed edge 
            doesn't exist in both directions.
        * **undirected** (List) - The undirected edges list within the input graph.
    """
    if len(edge_pairs) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    labels = -np.ones(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = np.array(list(map(list, edge_pairs)))

    # get directed edges
    edge_pairs = np.array(list(map(list, edge_pairs)))
    if signed_directed:
        directed_pos = (
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten() > 0).tolist()
        directed_neg = (
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten() < 0).tolist()
        inversed_pos = (
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten() > 0).tolist()
        inversed_neg = (
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten() < 0).tolist()
        undirected_pos = np.logical_and(directed_pos, inversed_pos)
        undirected_neg = np.logical_and(directed_neg, inversed_neg)
        undirected_pos_neg = np.logical_and(directed_pos, inversed_neg)
        undirected_neg_pos = np.logical_and(directed_neg, inversed_pos)

        directed_pos = list(map(tuple, edge_pairs[directed_pos].tolist()))
        directed_neg = list(map(tuple, edge_pairs[directed_neg].tolist()))
        inversed_pos = list(map(tuple, edge_pairs[inversed_pos].tolist()))
        inversed_neg = list(map(tuple, edge_pairs[inversed_neg].tolist()))
        undirected = np.logical_or(np.logical_or(np.logical_or(undirected_pos, undirected_neg), undirected_pos_neg), undirected_neg_pos)
        undirected = list(map(tuple, edge_pairs[np.array(undirected)].tolist()))

        edge_pairs = list(map(tuple, edge_pairs.tolist()))
        negative = np.array(
            list(set(edge_pairs) - set(directed_pos) - set(inversed_pos) - set(directed_neg) - set(inversed_neg)))
        directed_pos = np.array(list(set(directed_pos) - set(undirected)))
        inversed_pos = np.array(list(set(inversed_pos) - set(undirected)))
        directed_neg = np.array(list(set(directed_neg) - set(undirected)))
        inversed_neg = np.array(list(set(inversed_neg) - set(undirected)))

        directed = np.vstack([directed_pos, directed_neg])
        undirected = np.array(undirected)
        new_edge_pairs = directed
        new_edge_pairs = np.vstack([new_edge_pairs, new_edge_pairs[:, [1, 0]]])
        new_edge_pairs = np.vstack([new_edge_pairs, negative])

        labels = np.vstack([np.zeros((len(directed_pos), 1), dtype=np.int32),
                            np.ones((len(directed_neg), 1), dtype=np.int32)])

        labels = np.vstack([labels, 2 * np.ones((len(directed_pos), 1), dtype=np.int32),
                            3 * np.ones((len(directed_neg), 1), dtype=np.int32)])

        labels = np.vstack(
            [labels, 4*np.ones((len(negative), 1), dtype=np.int32)])

        label_weight = np.vstack([np.array(adj[directed_pos[:, 0], directed_pos[:, 1]]).flatten()[:, None],
                                np.array(adj[directed_neg[:, 0], directed_neg[:, 1]]).flatten()[:, None]])
        label_weight = np.vstack([label_weight, label_weight])
        label_weight = np.vstack(
            [label_weight, np.zeros((len(negative), 1), dtype=np.int32)])
        assert label_weight[labels==0].min() > 0
        assert label_weight[labels==1].max() < 0
        assert label_weight[labels==2].min() > 0
        assert label_weight[labels==3].max() < 0
        assert label_weight[labels==4].mean() == 0
    elif directed_graph:
        directed = (np.abs(
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()) > 0).tolist()
        inversed = (np.abs(
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten()) > 0).tolist()
        undirected = np.logical_and(directed, inversed)

        directed = list(map(tuple, edge_pairs[directed].tolist()))
        inversed = list(map(tuple, edge_pairs[inversed].tolist()))
        undirected = list(map(tuple, edge_pairs[undirected].tolist()))

        edge_pairs = list(map(tuple, edge_pairs.tolist()))
        negative = np.array(
            list(set(edge_pairs) - set(directed) - set(inversed)))
        directed = np.array(list(set(directed) - set(undirected)))
        inversed = np.array(list(set(inversed) - set(undirected)))

        new_edge_pairs = directed
        new_edge_pairs = np.vstack([new_edge_pairs, new_edge_pairs[:, [1, 0]]])
        new_edge_pairs = np.vstack([new_edge_pairs, negative])

        labels = np.zeros((len(directed), 1), dtype=np.int32)
        labels = np.vstack([labels, np.ones((len(directed), 1), dtype=np.int32)])
        labels = np.vstack(
            [labels, 2*np.ones((len(negative), 1), dtype=np.int32)])

        label_weight = np.array(adj[directed[:, 0], directed[:, 1]]).flatten()[:, None]
        label_weight = np.vstack([label_weight, label_weight])
        label_weight = np.vstack(
            [label_weight, np.zeros((len(negative), 1), dtype=np.int32)])
        assert abs(label_weight[labels==0]).min() > 0
        assert abs(label_weight[labels==1]).min() > 0
        assert label_weight[labels==2].mean() == 0
    else:
        undirected = []
        neg_edges = (
            np.abs(np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()) == 0)
        labels = np.ones(len(edge_pairs), dtype=np.int32)
        labels[neg_edges] = 2
        new_edge_pairs = edge_pairs
        label_weight = np.array(
            adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()
        labels[label_weight < 0] = 0
        if adj.data.min() < 0: # signed graph
            assert label_weight[labels==0].max() < 0
        if keep_negatives == False:
            assert label_weight[labels==1].min() > 0
        assert label_weight[labels==2].mean() == 0

    if task == 'existence':
        labels[labels == 1] = 0
        labels[labels == 2] = 1
        assert label_weight[labels == 1].mean() == 0
        if keep_negatives == False:
            assert abs(label_weight[labels == 0]).min() > 0
        

    return new_edge_pairs, labels.flatten(), label_weight.flatten(), undirected


def link_class_split_new(data: torch_geometric.data.Data, size: int = None, splits: int = 2, prob_test: float = 0.05,
                     prob_val: float = 0.05, task: str = 'existence', seed: int = 0, maintain_connect: bool = True,
                     ratio: float = 1.0, device: str = 'cpu', keep_negatives: bool = False) -> dict:
    r"""Get train/val/test dataset for the link prediction task. 
    Arg types:
        * **data** (torch_geometric.data.Data or DirectedData object) - The input dataset.
        * **prob_val** (float, optional) - The proportion of edges selected for validation (Default: 0.05).
        * **prob_test** (float, optional) - The proportion of edges selected for testing (Default: 0.15).
        * **splits** (int, optional) - The split size (Default: 2).
        * **size** (int, optional) - The size of the input graph. If none, the graph size is the maximum index of nodes plus 1 (Default: None).
        * **task** (str, optional) - The evaluation task: three_class_digraph (three-class link prediction); direction (direction prediction); existence (existence prediction); sign (sign prediction); four_class_signed_digraph (directed sign prediction); five_class_signed_digraph (directed sign and existence prediction) (Default: 'direction')
        * **seed** (int, optional) - The random seed for positve edge selection (Default: 0). Negative edges are selected by pytorch geometric negative_sampling.
        * **maintain_connect** (bool, optional) - If maintaining connectivity when removing edges for validation and testing. The connectivity is maintained by obtaining edges in the minimum spanning tree/forest first. These edges will not be removed for validation and testing (Default: True). 
        * **ratio** (float, optional) - The maximum ratio of edges used for dataset generation. (Default: 1.0)
        * **device** (int, optional) - The device to hold the return value (Default: 'cpu').
    Return types:
        * **datasets** - A dict include training/validation/testing splits of edges and labels. For split index i:
            * datasets[i]['graph'] (torch.LongTensor): the observed edge list after removing edges for validation and testing.
            * datasets[i]['train'/'val'/'testing']['edges'] (List): the edge list for training/validation/testing.
            * datasets[i]['train'/'val'/'testing']['label'] (List): the labels of edges:
                * If task == "existence": 0 (the directed edge exists in the graph), 1 (the edge doesn't exist). The undirected edges in the directed input graph are removed to avoid ambiguity.
                * If task == "direction": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists). The undirected edges in the directed input graph are removed to avoid ambiguity.
                * If task == "three_class_digraph": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists), 2 (the edge doesn't exist in both directions). The undirected edges in the directed input graph are removed to avoid ambiguity.
                * If task == "four_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                    1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                    3 (the edge of the reversed direction exists). 
                    The undirected edges in the directed input graph are removed to avoid ambiguity.
                
                * If task == "five_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                    1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                    3 (the edge of the reversed direction exists), 4 (the edge doesn't exist in both directions). 
                    The undirected edges in the directed input graph are removed to avoid ambiguity.
                
                * If task == "sign": 0 (negative edge), 1 (positive edge). This is the link sign prediction task for signed networks.
    """
    assert task in ["existence", "direction", "three_class_digraph", "four_class_signed_digraph", "five_class_signed_digraph", 
                    "sign", "weight_prediction"], "Please select a valid task from 'existence', 'direction', 'three_class_digraph', 'four_class_signed_digraph', 'five_class_signed_digraph', 'sign', and 'weight_prediction'!"
    weighted = False
    if task == "weight_prediction":
        task = "existence"
        weighted = True

    edge_index = data.edge_index.cpu()
    row, col = edge_index[0], edge_index[1]
    if size is None:
        size = int(max(torch.max(row), torch.max(col))+1)
    if not hasattr(data, "edge_weight"):
        data.edge_weight = torch.ones(len(row))
    if data.edge_weight is None:
        data.edge_weight = torch.ones(len(row))

    # we did not normalize data.A, we normalized only data.edge_weight - so use it
    #if hasattr(data, "A"):
    #    A = data.A.tocsr()
    #else:
    A = coo_matrix((data.edge_weight, (row, col)),
                       shape=(size, size), dtype=np.float32).tocsr()

    len_val = int(prob_val*len(row))
    len_test = int(prob_test*len(row))
    if task not in ["existence", "direction", 'three_class_digraph']:
        pos_ratio = (A>0).sum()/len(A.data)
        neg_ratio = 1 - pos_ratio
        len_val_pos = int(np.around(prob_val*len(row)*pos_ratio))
        len_val_neg = int(np.around(prob_val*len(row)*neg_ratio))
        len_test_pos = int(np.around(prob_test*len(row)*pos_ratio))
        len_test_neg = int(np.around(prob_test*len(row)*neg_ratio))

    undirect_edge_index = to_undirected(edge_index)
    neg_edges = negative_sampling(undirect_edge_index, num_neg_samples=len(
        edge_index.T), force_undirected=False).numpy().T
    neg_edges = map(tuple, neg_edges)
    neg_edges = list(neg_edges)

    all_edge_index = edge_index.T.tolist()
    A_undirected = to_scipy_sparse_matrix(undirect_edge_index)
    if maintain_connect:
        assert ratio == 1, "ratio should be 1.0 if maintain_connect=True"
        G = nx.from_scipy_sparse_array(
            A_undirected, create_using=nx.Graph, edge_attribute='weight')
        mst = list(tree.minimum_spanning_edges(
            G, algorithm="kruskal", data=False))
        all_edges = list(map(tuple, all_edge_index))
        mst_r = [t[::-1] for t in mst]
        nmst = list(set(all_edges) - set(mst) - set(mst_r))
        if len(nmst) < (len_val+len_test):
            raise ValueError(
                "There are no enough edges to be removed for validation/testing. Please use a smaller prob_test or prob_val.")
    else:
        mst = []
        nmst = edge_index.T.tolist()

    rs = np.random.RandomState(seed)
    datasets = {}

    max_samples = int(ratio*len(edge_index.T))+1
    assert ratio <= 1.0 and ratio > 0, "ratio should be smaller than 1.0 and larger than 0"
    assert ratio > prob_val + prob_test, "ratio should be larger than prob_val + prob_test"
    for ind in range(splits):
        rs.shuffle(nmst)
        rs.shuffle(neg_edges)

        if task in ["direction", 'three_class_digraph']:
            ids_test = nmst[:len_test]+neg_edges[:len_test]
            ids_val = nmst[len_test:len_test+len_val] + \
                neg_edges[len_test:len_test+len_val]
            if len_test+len_val < len(nmst):
                ids_train = nmst[len_test+len_val:max_samples] + \
                    mst+neg_edges[len_test+len_val:max_samples]
            else:
                ids_train = mst+neg_edges[len_test+len_val:max_samples]

            ids_test, labels_test, _, _ = undirected_label2directed_label(
                A, ids_test, task, True)
            ids_val, labels_val, _, _ = undirected_label2directed_label(
                A, ids_val, task, True)
            ids_train, labels_train, _, undirected_train = undirected_label2directed_label(
                A, ids_train, task, True)
        elif task == "existence":
            ids_test = nmst[:len_test]+neg_edges[:len_test]
            ids_val = nmst[len_test:len_test+len_val] + \
                neg_edges[len_test:len_test+len_val]
            if len_test+len_val < len(nmst):
                ids_train = nmst[len_test+len_val:max_samples] + \
                    mst+neg_edges[len_test+len_val:max_samples]
            else:
                ids_train = mst+neg_edges[len_test+len_val:max_samples]

            ids_test, labels_test, _, _ = undirected_label2directed_label(
                A, ids_test, task, False, keep_negatives=keep_negatives)
            ids_val, labels_val, _, _ = undirected_label2directed_label(
                A, ids_val, task, False, keep_negatives=keep_negatives)
            ids_train, labels_train, _, undirected_train = undirected_label2directed_label(
                A, ids_train, task, False, keep_negatives=keep_negatives)
            weights = A[ids_val[:, 0], ids_val[:, 1]]
            assert abs(weights[:, labels_val == 1]).mean() == 0
        elif task == 'sign':
            nmst = np.array(nmst)
            pos_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] > 0).squeeze()].tolist()
            neg_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] < 0).squeeze()].tolist()

            ids_test = np.array(pos_val_edges[:len_test_pos].copy() + neg_val_edges[:len_test_neg].copy() + \
                neg_edges[:len_test])
            ids_val = np.array(pos_val_edges[len_test_pos:len_test_pos+len_val_pos].copy() + \
                neg_val_edges[len_test_neg:len_test_neg+len_val_neg].copy() + \
                neg_edges[len_test:len_test+len_val])
            
            if len_test+len_val < len(nmst):
                ids_train = np.array(pos_val_edges[len_test_pos+len_val_pos:max_samples] + \
                    neg_val_edges[len_test_neg+len_val_neg:max_samples] + mst + neg_edges[len_test+len_val:max_samples])
            else:
                ids_train = mst+neg_edges[len_test+len_val:max_samples]

            ids_test, labels_test, _, _ = undirected_label2directed_label(
                A, ids_test, task, False, False)
            ids_val, labels_val, _, _ = undirected_label2directed_label(
                A, ids_val, task, False, False)
            ids_train, labels_train, _, undirected_train = undirected_label2directed_label(
                A, ids_train, task, False, False)
        else:
            nmst = np.array(nmst)
            pos_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] > 0).squeeze()].tolist()
            neg_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] < 0).squeeze()].tolist()

            ids_test = np.array(pos_val_edges[:len_test_pos].copy() + neg_val_edges[:len_test_neg].copy() + \
                neg_edges[:len_test])
            ids_val = np.array(pos_val_edges[len_test_pos:len_test_pos+len_val_pos].copy() + \
                neg_val_edges[len_test_neg:len_test_neg+len_val_neg].copy() + \
                neg_edges[len_test:len_test+len_val])
            
            if len_test+len_val < len(nmst):
                ids_train = np.array(pos_val_edges[len_test_pos+len_val_pos:max_samples] + \
                    neg_val_edges[len_test_neg+len_val_neg:max_samples] + mst + neg_edges[len_test+len_val:max_samples])
            else:
                ids_train = mst+neg_edges[len_test+len_val:max_samples]

            ids_test, labels_test, _, _ = undirected_label2directed_label(
                A, ids_test, task, True, True)
            ids_val, labels_val, _, _ = undirected_label2directed_label(
                A, ids_val, task, True, True)
            ids_train, labels_train, _, undirected_train = undirected_label2directed_label(
                A, ids_train, task, True, True)

        # convert back to directed graph
        if task in ['direction', 'sign']:
            ids_train = ids_train[labels_train < 2]
            #label_train_w = label_train_w[labels_train <2]
            labels_train = labels_train[labels_train < 2]

            ids_test = ids_test[labels_test < 2]
            #label_test_w = label_test_w[labels_test <2]
            labels_test = labels_test[labels_test < 2]

            ids_val = ids_val[labels_val < 2]
            #label_val_w = label_val_w[labels_val <2]
            labels_val = labels_val[labels_val < 2]
        elif task == 'four_class_signed_digraph':
            ids_train = ids_train[labels_train < 4]
            labels_train = labels_train[labels_train < 4]

            ids_test = ids_test[labels_test < 4]
            labels_test = labels_test[labels_test < 4]

            ids_val = ids_val[labels_val < 4]
            labels_val = labels_val[labels_val < 4]
            
        # set up the observed graph and weights after splitting
        observed_edges = -np.ones((len(ids_train), 2), dtype=np.int32)
        observed_weight = np.zeros((len(ids_train), 1), dtype=np.float32)

        direct = (
            np.abs(A[ids_train[:, 0], ids_train[:, 1]].data) > 0).flatten()
        observed_edges[direct, 0] = ids_train[direct, 0]
        observed_edges[direct, 1] = ids_train[direct, 1]
        observed_weight[direct, 0] = np.array(
            A[ids_train[direct, 0], ids_train[direct, 1]]).flatten()

        valid = (np.sum(observed_edges, axis=-1) >= 0)
        observed_edges = observed_edges[valid]
        observed_weight = observed_weight[valid]
        
        # add undirected edges back
        if len(undirected_train) > 0:
            undirected_train = np.array(undirected_train)
            observed_edges = np.vstack(
                (observed_edges, undirected_train))
            observed_weight = np.vstack((observed_weight, np.array(A[undirected_train[:, 0],
                                                                   undirected_train[:, 1]]).flatten()[:, None]))

        assert(len(edge_index.T) >= len(observed_edges)), 'The original edge number is {} \
            while the observed graph has {} edges!'.format(len(edge_index.T), len(observed_edges))

        datasets[ind] = {}
        datasets[ind]['graph'] = torch.from_numpy(
            observed_edges.T).long().to(device)
        datasets[ind]['weights'] = torch.from_numpy(
            observed_weight.flatten()).float().to(device)
        #print('----- Pesi ---')
        #print(datasets[ind]['weights'] )
        datasets[ind]['train'] = {}
        datasets[ind]['train']['edges'] = torch.from_numpy(
            ids_train).long().to(device)
        datasets[ind]['train']['label'] = torch.from_numpy(
            labels_train).long().to(device)
        #datasets[ind]['train']['weight'] = torch.from_numpy(label_train_w).float().to(device)

        datasets[ind]['val'] = {}
        datasets[ind]['val']['edges'] = torch.from_numpy(
            ids_val).long().to(device)
        datasets[ind]['val']['label'] = torch.from_numpy(
            labels_val).long().to(device)
        #datasets[ind]['val']['weight'] = torch.from_numpy(label_val_w).float().to(device)

        datasets[ind]['test'] = {}
        datasets[ind]['test']['edges'] = torch.from_numpy(
            ids_test).long().to(device)
        datasets[ind]['test']['label'] = torch.from_numpy(
            labels_test).long().to(device)
        #datasets[ind]['test']['weight'] = torch.from_numpy(label_test_w).float().to(device)

        if weighted == True:
             datasets[ind]['train']['label'] = torch.from_numpy(A[ids_train[:, 0], ids_train[:, 1]]).float().to(device)
             datasets[ind]['val']['label'] = torch.from_numpy(A[ids_val[:, 0], ids_val[:, 1]]).float().to(device)
             datasets[ind]['test']['label'] = torch.from_numpy(A[ids_test[:, 0], ids_test[:, 1]]).float().to(device)

    return datasets


def compute_scaled_normalized_Haar_laplacian(row, col, size, renormalize=True, lambda_max=2.0, edge_weight=None):
    if edge_weight is None:
        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)
    
    diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    
    #if renormalize:
    #    A += diag

    As = 0.5*(A + A.T)
    Aa = 0.5*(A - A.T)

    if renormalize:
        As += diag
    Hh = As + 1j*Aa
    d = np.array(abs(Hh).sum(axis=0))[0]
    d = np.power(d, -0.5)
    Dh = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    Hh = Dh.dot(Hh).dot(Dh)
    #Lh = diag - Hh

    #Lh = (2/lambda_max)*Lh - diag
    Lh = Hh
    return Lh
    
def compute_scaled_normalized_laplacian(row, col, size, renormalize=True, lambda_max=2.0, edge_weight=None):
    if edge_weight is None:
        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)
    
    diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    if renormalize:
        A += diag

    A_sym = 0.5*(A + A.T) # symmetrized adjacency

    # symmetric normalization
    d = np.array(A_sym.sum(axis=0))[0] # out degree
    d[d <= 0] = 1
    d = np.power(d, -0.5)
    D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    A_sym = D.dot(A_sym).dot(D)

    L = diag - A_sym
    L = (2.0/lambda_max)*L - diag
    return L


def cnv_sparse_mat_to_coo_tensor(sp_mat):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.complex64 or sp_mat.dtype == np.complex128:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.complex64, requires_grad=False)
    elif sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')


def intensityLaplacian(row, col, size, edge_weight=None, renormalize=True, lambda_max=2.0):
    if edge_weight is None:
        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)
    nb_nodes = size

    Acsr = A.tocsr()
    Acsc = A.tocsc()
    successors_gaw = np.zeros((nb_nodes, 1))
    predecessors_gaw = np.zeros((nb_nodes, 1))
    for x in range(size):
        succ_x = Acsr.getrow(x)
        s_weights = [s for s in succ_x.toarray().squeeze().tolist() if s > 0]
        if len(s_weights) > 0:
            successors_gaw[x] = st.gmean(s_weights)
        pred_x = Acsc.getcol(x)
        p_weights = [p for p in pred_x.toarray().squeeze().tolist() if p > 0]
        if len(p_weights) > 0:
            predecessors_gaw[x] = st.gmean(p_weights)


    graph = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    data = []
    rows = []
    cols = []
    mapping = {}
    k = 0
    for x in graph:
        successors_x = graph.successors(x)
        predecessors_x = graph.predecessors(x)
        for s in successors_x:
            pos1 = mapping.get((x, s))
            pos2 = mapping.get((s, x))
            i = max(predecessors_gaw[s], successors_gaw[x])
            if pos1 != None:
                data[pos1] = max(data[pos1], i[0])
            else:
                data.append(i[0])
                rows.append(x)
                cols.append(s)
                mapping[(x, s)] = k
                k += 1
            if pos2 != None:
                data[pos2] = max(data[pos2], i[0])
            else:
                data.append(i[0])
                rows.append(s)
                cols.append(x)
                mapping[(s, x)] = k
                k += 1
        for p in predecessors_x:
            pos1 = mapping.get((p, x))
            pos2 = mapping.get((x, p))
            i = max(predecessors_gaw[x], successors_gaw[p])
            if pos1 != None:
                data[pos1] = max(data[pos1], i[0])
            else:
                data.append(i[0])
                rows.append(p)
                cols.append(x)
                mapping[(p, x)] = k
                k += 1          
            if pos2 != None:
                data[pos2] = max(data[pos2], i[0])
            else:
                data.append(i[0])
                rows.append(x)
                cols.append(p)
                mapping[(x, p)] = k
                k += 1
    data = np.array(data)
    rows = np.array(rows)
    cols = np.array(cols)
    A = coo_matrix((data, (rows, cols)), shape=(nb_nodes, nb_nodes), dtype=np.float32)
    
    diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    if renormalize:
        A += diag

    # symmetric normalization
    d = np.array(A.sum(axis=0))[0] # out degree
    d[d <= 0] = 1
    d = np.power(d, -0.5)
    D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    A = D.dot(A).dot(D)

    L = diag - A
    L = (2.0/lambda_max)*L - diag
    return L


def intensityAdjacency(graph, graph_name, weights_normalization=None):
    if weights_normalization != None:
        graph_name = graph_name + '_' + weights_normalization
    try:
        A = sio.mmread(graph_name + '_intensities.mtx')
        return A
    except FileNotFoundError:
        nb_nodes = graph.number_of_nodes()
        successors_gaw = np.zeros((nb_nodes, 1))
        predecessors_gaw = np.zeros((nb_nodes, 1))
        for x in graph:
            succ_weights = []
            pred_weights = []
            successors_x = graph.successors(x)
            predecessors_x = graph.predecessors(x)
            for y in successors_x:
               succ_weights.append(graph[x][y]['weight'])
            for y in predecessors_x:
                pred_weights.append(graph[y][x]['weight'])
            successors_gaw[x] = 0
            if len(succ_weights) > 0:
                successors_gaw[x] = st.gmean(succ_weights)
            predecessors_gaw[x] = 0
            if len(pred_weights) > 0:
                predecessors_gaw[x] = st.gmean(pred_weights)

        data = []
        rows = []
        cols = []
        mapping = {}
        k = 0
        for x in graph:
            successors_x = graph.successors(x)
            predecessors_x = graph.predecessors(x)
            for s in successors_x:
                pos1 = mapping.get((x, s))
                pos2 = mapping.get((s, x))
                i = max(predecessors_gaw[s], successors_gaw[x])
                if pos1 != None:
                    data[pos1] = max(data[pos1], i[0])
                else:
                    data.append(i[0])
                    rows.append(x)
                    cols.append(s)
                    mapping[(x, s)] = k
                    k += 1
                if pos2 != None:
                    data[pos2] = max(data[pos2], i[0])
                else:
                    data.append(i[0])
                    rows.append(s)
                    cols.append(x)
                    mapping[(s, x)] = k
                    k += 1

            for p in predecessors_x:
                pos1 = mapping.get((p, x))
                pos2 = mapping.get((x, p))
                i = max(predecessors_gaw[x], successors_gaw[p])
                if pos1 != None:
                    data[pos1] = max(data[pos1], i[0])
                else:
                    data.append(i[0])
                    rows.append(p)
                    cols.append(x)
                    mapping[(p, x)] = k
                    k += 1          
                if pos2 != None:
                    data[pos2] = max(data[pos2], i[0])
                else:
                    data.append(i[0])
                    rows.append(x)
                    cols.append(p)
                    mapping[(x, p)] = k
                    k += 1

        data = np.array(data)
        rows = np.array(rows)
        cols = np.array(cols)
        adj_intensity = coo_matrix((data, (rows, cols)), shape=(nb_nodes, nb_nodes))
        sio.mmwrite(graph_name + '_intensities.mtx', adj_intensity)
        return adj_intensity
    

def negative_remove(data, double, constant):
    edge_index = data.edge_index
    row, col = edge_index[0], edge_index[1]
    size = int(max(torch.max(row), torch.max(col))+1)
    A = coo_matrix((data.edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32)
    mask = A.data<0
    A.data[mask]=0
    A.eliminate_zeros()
    if double:
        A.data = A.data*constant
    edge_index, weight = torch_geometric.utils.from_scipy_sparse_matrix(A)
    return edge_index, weight

def load_signed_real_data_no_negative(dataset: str='epinions', root:str = './tmp_data/', double : Optional[Callable] = False,
                            constant: Union[int,float]=None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                            train_size: Union[int,float]=None, val_size: Union[int,float]=None, 
                            test_size: Union[int,float]=None, seed_size: Union[int,float]=None,
                            train_size_per_class: Union[int,float]=None, val_size_per_class: Union[int,float]=None,
                            test_size_per_class: Union[int,float]=None, seed_size_per_class: Union[int,float]=None, 
                            seed: List[int]=[], data_split: int=10,
                            keep_negatives: bool=False) -> SignedData:
    """The function for real-world signed data downloading and convert to SignedData object.
    Arg types:
        * **dataset** (str, optional) - data set name (default: 'epinions').
        * **root** (str, optional) - The path to save the dataset (default: './').
        * **transform** (callable, optional) - A function/transform that takes in an \
            :obj:`torch_geometric.data.Data` object and returns a transformed \
            version. The data object will be transformed before every access. (default: :obj:`None`)
        * **pre_transform** (callable, optional) - A function/transform that takes in \
            an :obj:`torch_geometric.data.Data` object and returns a \
            transformed version. The data object will be transformed before \
            being saved to disk. (default: :obj:`None`)
        * **train_size** (int or float, optional) - The size of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size** (int or float, optional) - The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **train_size_per_class** (int or float, optional) - The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.  
        * **val_size_per_class** (int or float, optional) - The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size_per_class** (int or float, optional) - The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size_per_class** (int or float, optional) - The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.  
        * **seed** (An empty list or a list with the length of data_split, optional) - The random seed list for each data split.
        * **data_split** (int, optional) - number of splits (Default : 10)
    Return types:
        * **data** (Data) - The required data object.
    """
    if dataset.lower() in ['bitcoin_otc', 'bitcoin_alpha', 'slashdot', 'epinions']:
        data = SDGNN_real_data.SDGNN_real_data(root=root, name=dataset, transform=transform, pre_transform=pre_transform)._data
    elif dataset.lower() in ['sp1500', 'rainfall', 'sampson', 'wikirfa', 'ppi'] or dataset[:8].lower() == 'fin_ynet':
        data = SSSNET_real_data(name=dataset, root=root, transform=transform, pre_transform=pre_transform)[0]
    else:
        raise NameError('Please input the correct data set name instead of {}!'.format(dataset))
    if keep_negatives is False:
        edge, weight = negative_remove(data, double, constant) 
        data.edge_index = edge
        data.edge_weight = weight
    signed_dataset = SignedData(edge_index=data.edge_index, edge_weight=data.edge_weight, init_data=data)
    if train_size is not None or train_size_per_class is not None:
        signed_dataset.node_split(train_size=train_size, val_size=val_size, 
            test_size=test_size, seed_size=seed_size, train_size_per_class=train_size_per_class,
            val_size_per_class=val_size_per_class, test_size_per_class=test_size_per_class,
            seed_size_per_class=seed_size_per_class, seed=seed, data_split=data_split)
    return signed_dataset


def cheb_poly_sparse(A, K):
    K += 1
    N = A.shape[0]  # [N, N]
    #multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian = []
    multi_order_laplacian.append( coo_matrix( (np.ones(N), (np.arange(N), np.arange(N))), 
                                                    shape=(N, N), dtype=np.float32) )
    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian.append(A)
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian.append( 2.0 * A.dot(multi_order_laplacian[k-1]) - multi_order_laplacian[k-2] )

    return multi_order_laplacian

def hermitian_decomp_sparse(row, col, size, q = 0.25, norm = True, laplacian = True, max_eigen = 2, gcn_appr = False, edge_weight = None):
    if edge_weight is None:
        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)
    
    diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    if gcn_appr:
        A += diag

    A_sym = 0.5*(A + A.T) # symmetrized adjacency

    #print(A_sym)

    #new_arr_0 = A_sym.todense()[A_sym.todense()<0]
    #print(new_arr_0)
    #exit()
    if norm:
        d = np.array(abs(A_sym).sum(axis=0))[0] # out degree
        #print(d)
        d[d <= 0] = 1
        d = np.power(d, -0.5)
        #print(d)
        #exit()
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

    if laplacian:
        Theta = 2*np.pi*q*1j*(A - A.T) # phase angle array
        Theta.data = np.exp(Theta.data)
        if norm:
            D = diag
        else:
            d = np.sum(A_sym, axis = 0) # diag of degree array
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - Theta.multiply(A_sym) #element-wise

    if norm:
        L = (2.0/max_eigen)*L - diag

    return L


def get_specific(vector, device):
    vector = vector.tocoo()
    row = torch.from_numpy(vector.row).to(torch.long)
    col = torch.from_numpy(vector.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0).to(device)
    edge_weight = torch.from_numpy(vector.data).to(device)
    return edge_index, edge_weight

def get_Sign_Magnetic_Laplacian(edge_index: torch.LongTensor, gcn: bool, net_flow:bool, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  return_lambda_max: bool = False):
    r""" Computes our Sign Magnetic Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    
    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **normalization** (str, optional) - The normalization scheme for the magnetic Laplacian (default: :obj:`sym`) -
            1. :obj:`None`: No normalization :math:`\mathbf{L} = \mathbf{D} - \mathbf{H}^{\sigma}`
            
            2. :obj:`"sym"`: Symmetric normalization :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{H}^{\sigma}`
            \mathbf{D}^{-1/2}`
        
        * **dtype** (torch.dtype, optional) - The desired data type of returned tensor in case :obj:`edge_weight=None`. (default: :obj:`None`)
        * **num_nodes** (int, optional) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        * **return_lambda_max** (bool, optional) - Whether to return the maximum eigenvalue. (default: :obj:`False`)
    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = torch_geometric.utils.remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = torch_geometric.utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index.cpu()
    size = num_nodes

    A = coo_matrix((edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32)
    
    diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)


    if net_flow:
        A = flipping.new_adj(A)
        A_double = 0
    else:
        A_double =  antiparallel.antiparalell(A)
    
    if gcn:
        A += diag

    A_sym = 0.5*(A + A.T) # symmetrized adjacency
    operation = diag + A_double + (scipy.sparse.csr_matrix.sign(np.abs(A) - np.abs(A.T)))*1j
    
    deg = np.array(np.abs(A_sym).sum(axis=0))[0] # out degree
    if normalization is None:
        D = coo_matrix((deg, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - A_sym.multiply(operation) #element-wise
    elif normalization == 'sym':
        deg[deg == 0]= 1
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')]= 0
        D = coo_matrix((deg_inv_sqrt, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)
        L = diag - A_sym.multiply(operation)

        
    if not return_lambda_max: 
        edge_index, edge_weight= get_specific(L, device)
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index, edge_weight.real, edge_weight.imag
    

def __norm__(
        edge_index,
        gcn,
        net_flow,
        num_nodes: Optional[int],
        edge_weight: torch_geometric.typing.OptTensor,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None
    ):
        """
        Get  Sign-Magnetic Laplacian.
        
        Arg types:
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * num_nodes (int, Optional) - Node features.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * edge_index, edge_weight_real, edge_weight_imag (PyTorch Float Tensor) - Magnetic laplacian tensor: edge index, real weights and imaginary weights.
        """
        edge_index, edge_weight = torch_geometric.utils.remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight_real, edge_weight_imag = get_Sign_Magnetic_Laplacian(
            edge_index, gcn, net_flow, edge_weight, normalization, dtype, num_nodes  )
        lambda_max.to(edge_weight_real.device)

        edge_weight_real = (2.0 * edge_weight_real) / lambda_max
        edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)

        _, edge_weight_real = torch_geometric.utils.add_self_loops(
            edge_index, edge_weight_real, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight_real is not None

        edge_weight_imag = (2.0 * edge_weight_imag) / lambda_max
        edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)

        edge_index, edge_weight_imag = torch_geometric.utils.add_self_loops(
            edge_index, edge_weight_imag, fill_value=0, num_nodes=num_nodes )
        assert edge_weight_imag is not None
        return edge_index, edge_weight_real, edge_weight_imag


def process_magnetic_laplacian(edge_index: torch.LongTensor, gcn: bool, net_flow:bool, x_real: Optional[torch.Tensor] = None, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  num_nodes: Optional[int] = None,
                  lambda_max=None,
                  return_lambda_max: bool = False,
):
    if normalization != 'sym' and lambda_max is None:        
        _, _, _, lambda_max =  get_Sign_Magnetic_Laplacian(
        edge_index, gcn, edge_weight, None, return_lambda_max=True )

    if lambda_max is None:
        lambda_max = torch.tensor(2.0, dtype=x_real.dtype, device=x_real.device)
    if not isinstance(lambda_max, torch.Tensor):
        lambda_max = torch.tensor(lambda_max, dtype=x_real.dtype,
                                      device=x_real.device)
    assert lambda_max is not None
    node_dim = -2
    edge_index, norm_real, norm_imag = __norm__(edge_index, gcn, net_flow,
                                        x_real.size(node_dim),
                                         edge_weight, normalization,
                                         lambda_max, dtype=x_real.dtype)
    
    return edge_index, norm_real, norm_imag

def get_magnetic_signed_Laplacian(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  q: Optional[float] = 0.25,
                  return_lambda_max: bool = False, 
                  absolute_degree: bool = True):
    r""" Computes the magnetic signed Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.
    
    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **normalization** (str, optional) - The normalization scheme for the magnetic Laplacian (default: :obj:`sym`) -
            1. :obj:`None`: No normalization :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            
            2. :obj:`"sym"`: Symmetric normalization :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        
        * **dtype** (torch.dtype, optional) - The desired data type of returned tensor in case :obj:`edge_weight=None`. (default: :obj:`None`)
        * **num_nodes** (int, optional) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        * **q** (float, optional) - The value q in the paper for phase.
        * **return_lambda_max** (bool, optional) - Whether to return the maximum eigenvalue. (default: :obj:`False`)
        * **absolute_degree** (bool, optional) - Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    
    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic signed Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic signed Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic signed Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = torch_geometric.utils.remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = torch_geometric.utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    sym_abs_attr = torch.cat([torch.abs(edge_weight), torch.abs(edge_weight)], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr, sym_abs_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes, "add")

    row, col = edge_index_sym[0], edge_index_sym[1]
    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym/2

    if absolute_degree:
        edge_weight_abs_sym = edge_attr[:, 2]
        edge_weight_abs_sym = edge_weight_abs_sym/2
        deg = scatter_add(edge_weight_abs_sym, row, dim=0, dim_size=num_nodes)
    else:
        deg = scatter_add(torch.abs(edge_weight_sym), row, dim=0, dim_size=num_nodes) # absolute values for edge weights

    edge_weight_q = torch.exp(1j * 2* np.pi * q * edge_attr[:, 1])

    if normalization is None:
        # L = D_bar_sym - A_sym Hadamard \exp(i \Theta^{(q)}).
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index_sym, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight_sym * edge_weight_q, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = D_bar_sym^{-1/2} A_sym D_bar_sym^{-1/2} Hadamard \exp(i \Theta^{(q)}).
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight_sym * deg_inv_sqrt[col] * edge_weight_q

        # L = I - A_norm.
        edge_index, tmp = torch_geometric.utils.add_self_loops(edge_index_sym, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    if not return_lambda_max:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index, edge_weight.real, edge_weight.imag, lambda_max