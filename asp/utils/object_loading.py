from torch.utils.data import DataLoader

import asp.datasets
from asp.collate_fn.collate import Collator
from asp.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            drop_last = True
        else:
            drop_last = False

        # create and join datasets
        ds = params["dataset"]
        dataset = configs.init_obj(ds, asp.datasets)

        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
        else:
            raise Exception()

        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=Collator(),
            shuffle=shuffle, num_workers=num_workers, drop_last=drop_last
        )
        dataloaders[split] = dataloader

    return dataloaders
