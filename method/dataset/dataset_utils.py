import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision.transforms import v2 as transforms
from typing import Tuple

from method.dataset.dataset_classes import CelebA, CelebAHQ, MUFAC, IdentityUnlearningDataset


def get_transforms(args) -> Tuple[transforms.Compose, transforms.Compose]:
    if args.dataset in ("celeba", "celebahq"):
        args.size = 224
        args.mean = (0.5, 0.5, 0.5)
        args.std = (0.5, 0.5, 0.5)
        args.task = "multilabel"

        if "vit" in args.model:
            args.size = 224

        base_transform = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(args.mean, args.std),
        ]

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.size, antialias=None),
                transforms.RandomHorizontalFlip(),
                *base_transform,
            ]
        )
        test_transform = transforms.Compose(
            [
                *base_transform,
                transforms.Resize((args.size, args.size), antialias=None),
            ]
        )

    elif args.dataset == "mufac":
        args.size = 224
        args.mean = (0.5, 0.5, 0.5)
        args.std = (0.5, 0.5, 0.5)
        args.task = "classification"

        base_transform = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(args.mean, args.std),
        ]
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.size, antialias=None),
                transforms.RandomHorizontalFlip(),
                *base_transform,
            ]
        )

        test_transform = transforms.Compose(
            [
                *base_transform,
                transforms.Resize((args.size, args.size), antialias=None),
            ]
        )

    return train_transform, test_transform


def get_datasets(args, splits: dict) -> IdentityUnlearningDataset:
    """
    Returns a dataset manager:

    Returns:
    ------
    datasets: IdentityUnlearningDataset
    """
    train_transform, test_transform = get_transforms(args)

    datasets = {}

    if args.dataset == "celeba":
        train_dataset = CelebA(
            root=args.data_dir,
            split="all",
            target_type=["attr", "identity"],
            transform=train_transform,
            download=False,
        )
        test_dataset = CelebA(
            root=args.data_dir,
            split="all",
            target_type=["attr", "identity"],
            transform=test_transform,
            download=False,
        )

        args.num_classes = train_dataset.get_num_classes()
        args.criterion = "binary_cross_entropy"

        datasets = IdentityUnlearningDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_identities=args.num_identities,
            splits=splits,
        )

    elif args.dataset == "celebahq":
        train_dataset = CelebAHQ(args.data_dir, transform=train_transform)
        test_dataset = CelebAHQ(args.data_dir, transform=test_transform)

        args.num_classes = len(train_dataset.classes)
        
        if args.dataset == "celebahq":
            args.criterion = "binary_cross_entropy"
        else:
            args.criterion = None

        datasets = IdentityUnlearningDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_identities=args.num_identities,
            splits=splits,
        )

    elif args.dataset == "mufac":
        train_dataset = MUFAC(args.data_dir, transform=train_transform)
        test_dataset = MUFAC(args.data_dir, transform=test_transform)
        args.num_classes = len(train_dataset.classes)
        args.criterion = "cross_entropy"

        datasets = IdentityUnlearningDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_identities=args.num_identities,
            splits=splits,
        )

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    return datasets


class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)  # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)  # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch
