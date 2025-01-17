from typing import Optional, Callable
import os
import json

import torch
from torch_geometric.data import (InMemoryDataset, download_url, Data)


dataset_name_url_dic = {
    "ucsocial": "https://github.com/wave-zuo/SEA/blob/main/data/ucsocial/ucsocial.csv"
}


class ReadDataset(InMemoryDataset):
    def __init__(self, name: str,  root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.url = dataset_name_url_dic[name]
        self.root = root

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        _, _, filename = self.url.rpartition('/')
        return filename

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = []
        edge_weight = []
        edge_index = []
        node_map = {}
        with open(self.raw_paths[0], 'r') as f:
            for line in f:
                x = line.strip().split(',')
                assert len(x) == 3
                a, b = x[0], x[1]
                if a not in node_map:
                    node_map[a] = len(node_map)
                if b not in node_map:
                    node_map[b] = len(node_map)
                a, b = node_map[a], node_map[b]
                data.append([a, b])

                edge_weight.append(float(x[2]))

            edge_index = [[i[0], int(i[1])] for i in data]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            edge_weight = torch.FloatTensor(edge_weight)
        map_file = os.path.join(self.processed_dir, 'node_id_map.json')
        with open(map_file, 'w') as f:
            f.write(json.dumps(node_map))

        data = Data(edge_index=edge_index, edge_weight=edge_weight)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1
