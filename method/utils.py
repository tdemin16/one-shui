import numpy as np
import os
import sys
import random
import torch
import torch.distributed as dist
from typing import Tuple
from method.dataset.dataset_utils import DistributedEvalSampler
from method.metrics import compute_unlearning_metrics
from method.models import get_model


class FakeScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self):
        pass

    def step(self):
        pass


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


class LinearScheduler:
    def __init__(
        self, initial_value: float = 1, final_value: float = 0, n_iterations: int = 10
    ):
        self.initial_value = initial_value
        self.final_value = final_value
        self.n_iterations = n_iterations
        self.current_iteration = -1

    def step(self):
        assert self.current_iteration < self.n_iterations, "LinearScheduler is done"
        self.current_iteration += 1
        magnitude = (self.current_iteration) / self.n_iterations
        step = (self.initial_value - self.final_value) * magnitude
        return self.initial_value - step


def set_params(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def get_optimizer(model: torch.nn.Module, args, lr=None) -> torch.optim.Optimizer:
    lr = lr if lr is not None else args.lr
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer, args
) -> torch.optim.lr_scheduler.LRScheduler:
    if args.scheduler == "const":
        scheduler = FakeScheduler()
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.t_max
        )
    elif args.scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs
        )

    else:
        raise ValueError(f"Unknown scheduler {args.scheduler}")

    return scheduler


def get_warmup_scheduler(
    optimizer: torch.optim.Optimizer, len_dataloader: int, args
) -> torch.optim.lr_scheduler.LRScheduler:
    if args.warmup_epochs > 0:
        warmup_scheduler = WarmUpLR(optimizer, len_dataloader * args.warmup_epochs)
    else:
        warmup_scheduler = FakeScheduler()
    return warmup_scheduler


def get_criterion(criterion: str) -> torch.nn.Module | None:
    if criterion == "binary_cross_entropy":
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif criterion is None:
        return None
    else:
        raise ValueError(f"Unknown criterion {criterion}")


def get_sampler(
    dataset: torch.utils.data.Dataset,
    shuffle: bool,
    distributed: bool,
    device: int,
    world_size: int,
) -> torch.utils.data.Sampler:
    if distributed and not is_dist_avail_and_initialized():
        raise ValueError("Distributed sampler requires distributed mode")
    if shuffle:
        if distributed:
            return torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=device
            )
        else:
            return torch.utils.data.RandomSampler(dataset)
    else:
        if distributed:
            return DistributedEvalSampler(dataset, num_replicas=world_size, rank=device)
        else:
            return torch.utils.data.SequentialSampler(dataset)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    debug: bool = False,
    task: str = "multilabel",
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds = []
    labels = []
    for image, identity, target in dataloader:
        image = image.to(device)
        target = target.to(device)
        pred = model(image)
        
        preds.append(pred.cpu())
        labels.append(target.cpu())
        if debug:
            break
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    return preds, labels


def get_retrain_metrics(datasets, criterion, args):
    retrain_path = os.path.join(
        args.checkpoint_dir,
        args.dataset,
        args.model,
        f"retrain_{args.num_identities}_{args.seed}.pth",
    )
    if os.path.exists(retrain_path):
        checkpoint = torch.load(retrain_path, map_location="cuda:0", weights_only=False)
    else:
        raise ValueError(f"Checkpoint {retrain_path} not found")
    
    block_print()
    retrain_model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        size=args.size,
        pretrained=args.pretrained,
    )
    retrain_model.load_state_dict(checkpoint["state_dict"])
    retrain_model = retrain_model.to(args.device)
    retrain_model.eval()
    retrain_metrics = compute_unlearning_metrics(retrain_model, datasets, criterion, None, args)
    enable_print()
    return retrain_metrics


def freeze_norm_stats(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()


def sanitize_state_dict(state_dict):
    new_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_dict[key[7:]] = value
        else:
            new_dict[key] = value

    return new_dict


def print_info(args, model, dataloader):
    match args.method:
        case "pretrain":
            print("### Pretrain ###")
        case "retrain":
            print("### Retrain ###")
        case "scrub":
            print("### SCRUB ###")
        case "meta_unlearn":
            print("### Loss Learning ###")
        case "ssd":
            print("### Selective Synapse Dampening ###")
        case "lipschitz":
            print("### Lipschitz Regularization ###")
        case "pgu":
            print("### Projected Gradient Unlearning ###")
        case _:
            raise ValueError(f"Unknown method {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Arguments\n{args}")
    print(
        f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M"
    )
    print(f"Number of Images: {len(dataloader.dataset)}")
    print(f"Number of Batches: {len(dataloader)}")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def gather_tensor(tensor, args):
    if args.distributed:
        tensor_list = [None for _ in range(args.world_size)]
        dist.all_gather_object(tensor_list, tensor)
        tensor_list = [t.to(args.device) for t in tensor_list]
        tensor = torch.cat(tensor_list, dim=0)
    return tensor


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.device = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
        args.device = args.rank % torch.cuda.device_count()
        if args.world_size == 1:
            args.distributed = False
            args.gpu_name = torch.cuda.get_device_name(0)
            torch.cuda.set_device(args.device)
            return
    else:
        print("Not using distributed mode")
        args.rank = 0
        args.world_size = 1
        args.device = 0
        args.distributed = False
        args.gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.set_device(args.device)
        return
    args.distributed = True

    torch.cuda.set_device(args.device)
    args.dist_backend = "nccl"
    print("| distributed init (rank {})".format(args.rank), flush=True)
    dist.init_process_group(
        backend=args.dist_backend, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)

    args.gpu_name = torch.cuda.get_device_name(0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    Disable printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__
