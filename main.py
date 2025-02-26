import copy
import itertools
import os
import torch
import wandb
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from method import utils
from method.approaches import (
    MetaUnlearn,
    SelectiveSynapseDampening,
    LipschitzRegularization,
    ProjectedGradientUnlearning,
    SCRUB,
    BadTeacher,
)
from method.configs import parse_args
from method.configs.const import get_const
from method.dataset import get_datasets
from method.engine import train_and_eval, evaluate
from method.metrics import compute_unlearning_metrics
from method.models import get_model


def pretrain(model: torch.nn.Module, datasets, run, args):
    assert args.world_size == 1, "Pretraining is not compatible with distributed training"
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Not training {name}")

    train_dataset = datasets.get_train_data()
    val_dataset = datasets.get_val_data()

    partial_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    train_loader = partial_loader(train_dataset, shuffle=True)
    val_loader = partial_loader(val_dataset, shuffle=False)

    optimizer = utils.get_optimizer(model, args)
    scheduler = utils.get_scheduler(optimizer, args)
    warmup_scheduler = utils.get_warmup_scheduler(optimizer, len(train_loader), args)

    criterion = utils.get_criterion(args.criterion)

    utils.print_info(args, model, train_loader)

    best_model, last_model = train_and_eval(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        clip_grad=args.clip_grad,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        debug=args.debug,
        run=run,
        task=args.task,
        evaluate_every=args.evaluate_every,
    )

    best_model.eval()
    utils.set_params(best_model, requires_grad=False)
    compute_unlearning_metrics(best_model, datasets, criterion, run, args)

    return best_model, last_model


def retrain(model, datasets, run, args):
    assert args.world_size == 1, "Retraining is not compatible with distributed training"
    retain_data = datasets.get_unlearning_data(train=True)["retain"]
    val_dataset = datasets.get_val_data()

    partial_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    retain_loader = partial_loader(retain_data, shuffle=True)
    val_loader = partial_loader(val_dataset, shuffle=False)

    optimizer = utils.get_optimizer(model, args)
    scheduler = utils.get_scheduler(optimizer, args)
    warmup_scheduler = utils.get_warmup_scheduler(optimizer, len(retain_loader), args)

    criterion = utils.get_criterion(args.criterion)

    utils.print_info(args, model, retain_loader)

    best_model, last_model = train_and_eval(
        model=model,
        train_dataloader=retain_loader,
        test_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        clip_grad=args.clip_grad,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        debug=args.debug,
        run=run,
        task=args.task,
        evaluate_every=args.evaluate_every,
    )

    best_model.eval()
    utils.set_params(best_model, requires_grad=False)
    compute_unlearning_metrics(best_model, datasets, criterion, run, args)

    return best_model, last_model


def scrub(model, datasets, run, args):
    assert args.world_size == 1, "SCRUB is not compatible with distributed training"

    unlearning_datasets = datasets.get_unlearning_data(train=args.use_train_aug)
    retain_dataset = unlearning_datasets["retain"]
    forget_dataset = unlearning_datasets["forget"]
    val_dataset = datasets.get_val_data(train=False)

    partial_loader = partial(
        torch.utils.data.DataLoader,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    retain_loader = partial_loader(retain_dataset, batch_size=args.batch_size_retain, shuffle=True)
    forget_loader = partial_loader(forget_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = partial_loader(val_dataset, batch_size=args.batch_size, shuffle=False)

    utils.print_info(args, model, retain_loader)

    model = model.to(args.device)
    criterion = utils.get_criterion(args.criterion)
    scrub = SCRUB(model, criterion, args.debug, run, args, args.task)
    model = scrub.unlearn_model(retain_loader, forget_loader, val_loader, args)

    model.eval()
    utils.set_params(model, requires_grad=False)
    compute_unlearning_metrics(model, datasets, criterion, run, args)

    return model, None


def bad_teacher(model, datasets, run, args):
    assert args.world_size == 1, "Bad Teacher is not compatible with distributed training"

    unlearning_datasets = datasets.get_unlearning_data(train=args.use_train_aug)
    retain_dataset = unlearning_datasets["retain"]
    forget_dataset = unlearning_datasets["forget"]
    val_dataset = datasets.get_val_data(train=False)

    dumb_teacher = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        size=args.size,
        pretrained=False,
    )

    model = model.to(args.device)
    dumb_teacher = dumb_teacher.to(args.device)

    bad_t = BadTeacher(
        model=model,
        dumb_teacher=dumb_teacher,
        debug=args.debug,
        run=run,
        args=args,
    )
    unlearned_model = bad_t.unlearn_model(retain_dataset, forget_dataset, val_dataset, args)

    unlearned_model.eval()
    utils.set_params(unlearned_model, requires_grad=False)
    compute_unlearning_metrics(
        unlearned_model, datasets, utils.get_criterion(args.criterion), run, args
    )

    return unlearned_model, None


def meta_unlearn(model, datasets, run, args):
    support_dataset = datasets.get_unlearning_data()["support"]
    support_dataloader = torch.utils.data.DataLoader(
        support_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    utils.print_info(args, model, support_dataloader)

    meta_unlearn = MetaUnlearn(
        model=model,
        in_features=args.num_classes,  # check
        task_criterion=utils.get_criterion(args.criterion),
        identities=datasets.TRAIN,
        epochs=args.epochs,
        use_retain=args.use_retain,
        forget_epochs=args.forget_epochs,
        forget_loss=args.forget_loss,
        use_accs=args.use_accs,
        debug=args.debug,
        run=run,
        args=args,
        task=args.task,
    )
    meta_unlearn.train_loss(datasets=datasets, args=args)

    unlearned_model = meta_unlearn.unlearn_model(support_dataloader, args)
    unlearned_model.eval()
    criterion = utils.get_criterion(args.criterion)

    try:
        retrain_metrics = utils.get_retrain_metrics(datasets, criterion, args)
    except ValueError:
        retrain_metrics = None

    compute_unlearning_metrics(unlearned_model, datasets, criterion, run, args, retrain_metrics)

    return unlearned_model, None


def ssd(model, datasets, run, args):
    assert args.world_size == 1, "SSD is not compatible with distributed training"
    support_dataset = datasets.get_unlearning_data(train=False)["support"]
    train_dataset = datasets.get_train_data(train=False)

    partial_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    support_loader = partial_loader(support_dataset)
    train_loader = partial_loader(train_dataset)

    criterion = utils.get_criterion(args.criterion)
    optimizer = utils.get_optimizer(model, args)

    utils.print_info(args, model, support_loader)

    model.eval()

    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "dampening_constant": args.ssd_dampening_constant,
        "selection_weighting": args.ssd_selection_weighting,
    }

    ssd = SelectiveSynapseDampening(model, criterion, optimizer, args.device, parameters)
    support_importance = ssd.calc_importance(support_loader, args.device)
    train_importance = ssd.calc_importance(train_loader, args.device)
    ssd.modify_weight(train_importance, support_importance)

    utils.set_params(model, requires_grad=False)
    criterion = utils.get_criterion(args.criterion)

    try:
        retrain_metrics = utils.get_retrain_metrics(datasets, criterion, args)
    except ValueError:
        retrain_metrics = None

    compute_unlearning_metrics(model, datasets, criterion, run, args, retrain_metrics)

    return model, None


def lipschitz(model, datasets, run, args):
    assert args.world_size == 1, "Lipschitz is not compatible with distributed training"
    support_dataset = datasets.get_unlearning_data(train=False)["support"]

    support_loader = torch.utils.data.DataLoader(
        support_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    criterion = utils.get_criterion(args.criterion)
    utils.print_info(args, model, support_loader)

    parameters = {
        "learning_rate": args.lr,
        "n_samples": args.lipschitz_n_samples,
        "lipschitz_weighting": args.lipschitz_weighting,
    }

    jit = LipschitzRegularization(model, parameters=parameters)
    jit.modify_weight(support_loader)

    model.eval()
    utils.set_params(model, requires_grad=False)

    try:
        retrain_metrics = utils.get_retrain_metrics(datasets, criterion, args)
    except ValueError:
        retrain_metrics = None

    compute_unlearning_metrics(model, datasets, criterion, run, args, retrain_metrics)

    return model, None


def pgu(model, datasets, run, args):
    assert args.world_size == 1, "PGU is not compatible with distributed training"
    train_dataset = datasets.get_train_data(train=True)
    test_dataset = datasets.get_test_data()
    support_dataset = datasets.get_unlearning_data(train=args.use_train_aug)["support"]

    partial_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    train_loader = partial_loader(train_dataset, shuffle=False)
    test_loader = partial_loader(test_dataset, shuffle=False)
    support_loader = partial_loader(support_dataset, shuffle=True)

    criterion = utils.get_criterion(args.criterion)

    utils.print_info(args, model, support_loader)

    pgu = ProjectedGradientUnlearning(gamma=args.pgu_gamma, epochs=args.epochs)
    pgu.compute_CGS(
        model=model,
        train_dataloader=train_loader,
        forget_dataloader=support_loader,
        device=args.device,
    )
    pgu.modify_weight(
        model=model,
        forget_dataloader=support_loader,
        test_dataloader=test_loader,
        lr=args.lr,
        device=args.device,
        task=args.task,
    )

    model.eval()
    utils.set_params(model, requires_grad=False)

    try:
        retrain_metrics = utils.get_retrain_metrics(datasets, criterion, args)
    except ValueError:
        retrain_metrics = None

    compute_unlearning_metrics(model, datasets, criterion, run, args, retrain_metrics)

    return model, None


def main(args):
    utils.init_distributed_mode(args)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    utils.seed_everything(args.seed)

    run = None
    if args.wandb and args.device == 0:
        mode = "offline" if args.offline else "online"
        run = wandb.init(
            name=args.name,
            project=args.project,
            entity=args.entity,
            config=args,
            mode=mode,
        )

    splits = None
    state_dict = None
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        args.dataset,
        args.model,
        f"pretrain_{args.num_identities}_{args.seed}.pth",
    )
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
        splits = checkpoint["splits"]
        state_dict = checkpoint["state_dict"]

    elif args.method != "pretrain" and args.method != "retrain":
        raise ValueError(f"Checkpoint {checkpoint_path} not found")

    datasets = get_datasets(args, splits=splits)

    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        size=args.size,
        pretrained=args.pretrained,
    )
    if state_dict is not None and args.method not in ("pretrain", "retrain"):
        model.load_state_dict(state_dict)

    model = model.to(args.device)

    if args.distributed:
        model = DDP(model, device_ids=[args.device], output_device=args.device)

    if args.method == "pretrain":
        best_model, last_model = pretrain(model, datasets, run, args)
    elif args.method == "retrain":
        best_model, last_model = retrain(model, datasets, run, args)
    elif args.method == "scrub":
        best_model, last_model = scrub(model, datasets, run, args)
    elif args.method == "bad_teacher":
        best_model, last_model = bad_teacher(model, datasets, run, args)
    elif args.method == "test_loss":
        best_model, last_model = test_loss(model, datasets, run, args)
    elif args.method == "meta_unlearn":
        best_model, last_model = meta_unlearn(model, datasets, run, args)
    elif args.method == "ssd":
        best_model, last_model = ssd(model, datasets, run, args)
    elif args.method == "lipschitz":
        best_model, last_model = lipschitz(model, datasets, run, args)
    elif args.method == "pgu":
        best_model, last_model = pgu(model, datasets, run, args)
    else:
        raise ValueError(f"Unknown method {args.method}")

    if run is not None:
        run.finish()

    if args.save and args.device == 0:
        path_dir = os.path.join(args.store_dir, args.dataset, args.model)

        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        # store_dir/dataset/model/method_numidenities_seed.pth
        if args.name == "test":
            name = f"{args.method}_{args.num_identities}_{args.seed}"
        else:
            name = args.name + f"_{args.seed}"
        checkpoint_path = os.path.join(path_dir, f"{name}.pth")

        print(f"Saving model to {checkpoint_path}")

        # store args for future evaluation and model
        checkpoint = {
            "args": args,
            "splits": datasets.get_splits(),
            "state_dict": best_model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    args = parse_args()

    main(args)
