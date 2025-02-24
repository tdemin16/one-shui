import torch
import torchmetrics
import warnings

from functools import partial
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

from method import utils
from method.dataset.dataset_classes import IdentityUnlearningDataset


def mean_average_precision(logits: torch.Tensor, targets: torch.Tensor):
    with warnings.catch_warnings(action="ignore"):
        logits = torch.nn.functional.sigmoid(logits)
        return torchmetrics.functional.average_precision(logits, targets, task="multilabel", num_labels=logits.size(1))


def evaluate_after_unlearning(
    model: torch.nn.Module,
    datasets: IdentityUnlearningDataset,
    criterion: torch.nn.Module,
    args,
):
    from method.engine import evaluate  # avoid circular import

    # get retain, forget, and test data
    unlearning_data = datasets.get_unlearning_data(train=False)
    unlearning_data.update({"val": datasets.get_val_data()})
    unlearning_data.update({"test": datasets.get_test_data()})

    partial_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=32,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    retain_sampler = utils.get_sampler(
        unlearning_data["retain"],
        shuffle=False,
        distributed=args.distributed,
        device=args.device,
        world_size=args.world_size,
    )
    forget_sampler = utils.get_sampler(
        unlearning_data["forget"],
        shuffle=False,
        distributed=args.distributed,
        device=args.device,
        world_size=args.world_size,
    )
    val_sampler = utils.get_sampler(
        unlearning_data["val"],
        shuffle=False,
        distributed=args.distributed,
        device=args.device,
        world_size=args.world_size,
    )
    test_sampler = utils.get_sampler(
        unlearning_data["test"],
        shuffle=False,
        distributed=args.distributed,
        device=args.device,
        world_size=args.world_size,
    )

    retain_loader = partial_loader(unlearning_data["retain"], sampler=retain_sampler)
    forget_loader = partial_loader(unlearning_data["forget"], sampler=forget_sampler)
    val_loader = partial_loader(unlearning_data["val"], sampler=val_sampler)
    test_loader = partial_loader(unlearning_data["test"], sampler=test_sampler)

    # evaluate model on retain, forget, and test set
    retain_stats = evaluate(
        model, retain_loader, criterion, args.device, args.debug, args.task, args, return_losses=True
    )
    forget_stats = evaluate(
        model, forget_loader, criterion, args.device, args.debug, args.task, args, return_losses=True
    )
    val_stats = evaluate(model, val_loader, criterion, args.device, args.debug, args.task, args, return_losses=True)
    test_stats = evaluate(model, test_loader, criterion, args.device, args.debug, args.task, args, return_losses=True)

    return (
        retain_stats["loss"],
        retain_stats["acc"],
        retain_stats["losses"],
        forget_stats["loss"],
        forget_stats["acc"],
        forget_stats["losses"],
        val_stats["loss"],
        val_stats["acc"],
        val_stats["losses"],
        test_stats["loss"],
        test_stats["acc"],
        test_stats["losses"],
    )


def compute_roc_auc(y_pred, y_true):
    roc = roc_curve(y_true, y_pred)[:2]
    auc = roc_auc_score(y_true, y_pred)
    return roc, auc


def compute_tow(appx, ref):
    return 1 - abs(appx - ref) / 100


@torch.no_grad()
def compute_loss(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, args):
    model.eval()
    losses = []
    for image, identity, target in dataloader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = image.to(device=args.device, dtype=torch.bfloat16)
            target = target.to(args.device)
            output = model(image)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target.float(), reduction="none").mean(
                dim=-1
            )

        losses.append(loss)

        if args.debug:
            break

    return torch.cat(losses, dim=0)


def compute_basic_mia(retain_losses, forget_losses, val_losses, test_losses):
    train_loss = torch.cat((retain_losses, val_losses), dim=0).unsqueeze(1).cpu().numpy()
    train_target = torch.cat((torch.ones(retain_losses.size(0)), torch.zeros(val_losses.size(0))), dim=0).numpy()
    test_loss = torch.cat((forget_losses, test_losses), dim=0).unsqueeze(1).cpu().numpy()
    test_target = torch.cat((torch.ones(forget_losses.size(0)), torch.zeros(test_losses.size(0)))).cpu().numpy()
    
    best_auc = 0
    best_acc = 0
    for n_est in [20, 50, 100]:
        for criterion in ['gini', 'entropy']:
            mia_model = RandomForestClassifier(n_estimators=n_est, criterion=criterion, n_jobs=8, random_state=0)
            mia_model.fit(train_loss, train_target)
            
            y_hat = mia_model.predict_proba(test_loss)[:, 1]
            auc = roc_auc_score(test_target, y_hat) * 100

            y_hat = mia_model.predict(forget_losses.unsqueeze(1).cpu().numpy()).mean()
            acc = (1 - y_hat) * 100

            if acc > best_acc:
                best_acc = acc
                best_auc = auc

    return best_auc, best_acc


def compute_unlearning_metrics(
    model: torch.nn.Module,
    datasets: IdentityUnlearningDataset,
    criterion: torch.nn.Module,
    run,
    args,
    retrain_metrics: dict | None = None,
) -> dict:
    """
    Compute metrics for class forgetting experiment
    """
    (
        retain_loss,
        retain_acc,
        retain_losses,
        forget_loss,
        forget_acc,
        forget_losses,
        val_loss,
        val_acc,
        val_losses,
        test_loss,
        test_acc,
        test_losses,
    ) = evaluate_after_unlearning(model, datasets, criterion, args)

    if isinstance(retain_acc, torch.Tensor):
        retain_acc = retain_acc.cpu().item()
        forget_acc = forget_acc.cpu().item()
        test_acc = test_acc.cpu().item()

    tow = -1
    if retrain_metrics is not None:
        tow = (
            compute_tow(retain_acc, retrain_metrics["retain/mAP"])
            * compute_tow(forget_acc, retrain_metrics["forget/mAP"])
            * compute_tow(test_acc, retrain_metrics["test/mAP"])
            * 100
        )

    mia_auc, mia_acc = compute_basic_mia(retain_losses, forget_losses, val_losses, test_losses)

    task_metrics = {
        "retain/loss": retain_loss,
        "retain/mAP": retain_acc,
        "forget/loss": forget_loss,
        "forget/mAP": forget_acc,
        "test/loss": test_loss,
        "test/mAP": test_acc,
        "mia/auc": mia_auc,
        "mia/acc": mia_acc,
    }
    if tow != -1:
        task_metrics["tow"] = tow

    metric = "mAP" if args.task == "multilabel" else "accuracy"
    print(
        f"| Retain Loss: {retain_loss:.4f} | Retain {metric}: {retain_acc:.2f} | Forget Loss: {forget_loss:.4f} | Forget {metric}: {forget_acc:.2f} | Test Loss: {test_loss:.4f} | Test {metric}: {test_acc:.2f} | MIA AUC: {mia_auc:.2f} | MIA Acc: {mia_acc:.2f} |",
        end="",
    )
    if tow != -1:
        print(f" ToW: {tow:.2f} |")
    else:
        print()

    if run is not None:
        log_dict = {
            "retain/loss": retain_loss,
            "retain/mAP": retain_acc,
            "forget/loss": forget_loss,
            "forget/mAP": forget_acc,
        }
        if tow != -1:
            log_dict["tow"] = tow

        # avoid logging twice the same quantity
        if args.method not in ["pretrain", "retrain"]:
            log_dict.update({"test/loss": test_loss, "test/mAP": test_acc})
        run.log(log_dict)

    return task_metrics
