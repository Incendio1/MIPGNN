import os.path as osp
from torch_geometric.datasets import Planetoid, WebKB, Actor
import torch_geometric.transforms as T
import torch
import numpy as np

import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
from torch_sparse import coalesce

from torch_geometric.data import Data, InMemoryDataset, download_url


class WikipediaNetwork(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')

    def __init__(self, root: str, name: str, geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self):
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0], 'r') as f:
                data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

            with open(self.raw_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                f = np.load(filepath)
                train_masks += [torch.from_numpy(f['train_mask'])]
                val_masks += [torch.from_numpy(f['val_mask'])]
                test_masks += [torch.from_numpy(f['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = torch.from_numpy(data['features']).to(torch.float)
            edge_index = torch.from_numpy(data['edges']).to(torch.long)
            edge_index = edge_index.t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            y = torch.from_numpy(data['target']).to(torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def get_dataset(name, normalize_features=False, transform=None):
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = get_planetoid_dataset(name, normalize_features)
        return dataset
    elif name in ['Wisconsin', 'Cornell', 'Texas']:
        dataset = get_WebKB_dataset(name, normalize_features)
        return dataset
    elif name in ['Actor']:
        dataset = get_Actor_dataset(name, normalize_features)
        return dataset
    elif name in ['Chameleon']:
        dataset = get_Wikipedia_dataset(name, normalize_features)
        return dataset


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_WebKB_dataset(name, normalize_features = False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WebKB(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dataset.data.y = dataset.data.y.long()   # not sure why here the data.y is not of long type as planetoid

    return dataset


# chameleon
def get_Wikipedia_dataset(name, normalize_features = False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WikipediaNetwork(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dataset.data.y = dataset.data.y.long()   # not sure why here the data.y is not of long type as planetoid

    return dataset


def get_Actor_dataset(name, normalize_features = False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Actor(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dataset.data.y = dataset.data.y.long()   # not sure why here the data.y is not of long type as planetoid

    return dataset


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1

    return mask


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def random_full_splits(data, num_classes, split, name):
    # Set new fixed random planetoid splits:

    indices = []
    train_indices = []
    val_indices = []
    test_indices = []

    spliter = np.load('./splits/' + name + '_split_0.6_0.2_' + str(split) + '.npz')    # split contains 10 fixed full-supervised setting data

    data.train_mask = torch.tensor(spliter['train_mask']).type(torch.bool)
    data.val_mask = torch.tensor(spliter['val_mask']).type(torch.bool)
    data.test_mask = torch.tensor(spliter['test_mask']).type(torch.bool)
    return data


def index_to_mask_1(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


# %60 %20
def random_planetoid_splits_1(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask_1(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask_1(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask_1(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask_1(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask_1(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask_1(rest_index, size=data.num_nodes)
    return data