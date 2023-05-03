from easydict import EasyDict
from typing import List
import numpy as np
import torch
from vision_base.utils.builder import build

def find_shared_keys(batch):
    lists_of_keys = [
        list(item.keys()) for item in batch
    ]
    shared_keys = set(lists_of_keys[0])
    for keylist in lists_of_keys[1:]:
        shared_keys = shared_keys.intersection(set(keylist))
    return list(shared_keys)

def collate_fn(batch):
    collated_data = {}
    shared_keys = find_shared_keys(batch)
    for key in shared_keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated_data[key] = torch.stack([item[key] for item in batch], dim=0)
        elif isinstance(batch[0][key], np.ndarray):
            collated_data[key] = torch.stack([torch.from_numpy(item[key]) for item in batch], dim=0)
        else:
            collated_data[key] = [item[key] for item in batch]

    return collated_data


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_list:List[EasyDict], **common_keywords):
        super(ConcatDataset, self).__init__()
        self.children:List = []
        for item in cfg_list:
            tmp = common_keywords.copy()
            tmp.update(item)
            self.children.append(build(**tmp))

        # build seperator based each other
        seperator = [0]
        for child in self.children[0:-1]:
            seperator.append(seperator[-1] + len(child))
        self.seperator = np.array(seperator)
        self.total_length = self.seperator[-1] + len(self.children[-1])

    def __len__(self):
        return self.total_length

    def _determine_index(self, index):
        child_index = np.searchsorted(self.seperator, index, side='right') - 1
        index_for_child = index - self.seperator[child_index]
        return child_index, index_for_child

    def __getitem__(self, index):
        child_index, index_for_child = self._determine_index(index)
        return self.children[child_index][index_for_child]
