import copy
import math
import torch
from torch.nn import functional as F
from typing import Iterable
from tqdm import tqdm

from method.metrics import mean_average_precision
from method import utils


def train(
    model: torch.nn.Module,
    train_dataloader: Iterable,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad: float,
    warmup_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    warmup_epochs: int,
    device: torch.device,
    debug: bool,
    task: str,
):

    model.train()
    losses = []
    accuracies = []
    id_losses = []
    id_accuracies = []

    scaler = torch.cuda.amp.GradScaler()

    # init progress bar
    pbar = tqdm(train_dataloader, total=len(train_dataloader), leave=False, dynamic_ncols=True)
    for image, identity, target in pbar:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = image.to(device=device, dtype=torch.bfloat16)
            target = target.to(device)
            
            output = model(image)

            if task == "multilabel":
                loss = criterion(output, target.float())
            elif task == "classification":
                loss = criterion(output, target)
            else:
                raise ValueError(f"Unknown criterion {criterion}")
        losses.append(loss.item())

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            exit()

        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # compute mAP for multilabel
        if task == "multilabel":
            acc = mean_average_precision(output, target) * 100
            accuracies.append(acc)

        elif task == "classification":
            acc = (output.argmax(dim=-1) == target).float().mean().item() * 100
            accuracies.append(acc)

        else:
            raise ValueError(f"Unknown task {task}")

        metric = "mAP" if task == "multilabel" else "Acc"
        train_loss = sum(losses) / len(losses)
        train_acc = sum(accuracies) / len(accuracies)

        pbar_description = f'| Lr: {optimizer.param_groups[0]["lr"]:.4f} | Loss: {train_loss:.4f} | {metric}: {train_acc:.2f} |'

        pbar.set_description(pbar_description)

        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()

        if debug:
            break

    train_loss = torch.tensor(losses).mean().item()
    accuracies = torch.tensor(accuracies).mean().item()
    train_stats = {"loss": train_loss, "acc": train_acc}

    if len(id_losses) > 0:
        id_losses = torch.tensor(id_losses).mean().item()
        id_accuracies = torch.tensor(id_accuracies).mean().item()
        train_stats["id_loss"] = id_losses
        train_stats["id_acc"] = id_accuracies

    return train_stats


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_dataloader: Iterable,
    criterion: torch.nn.Module,
    device: torch.device,
    debug: bool,
    task: str,
    args=None,
    return_losses=False,
):
    model.eval()
    preds = []
    targets = []
    losses = []
    full_losses = []

    for image, identity, target in tqdm(test_dataloader, leave=False, dynamic_ncols=True):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = image.to(device=device, dtype=torch.bfloat16)
            target = target.to(device)
            output = model(image)

            # if output is task_prediction, identity_prediction, then take only task_prediction
            if isinstance(output, tuple):
                output = output[0]

            if task == "multilabel":
                loss = criterion(output, target.float())
                full_loss = F.binary_cross_entropy_with_logits(
                    output, target.float(), reduction="none"
                ).mean(dim=-1)
            elif task == "classification":
                loss = criterion(output, target)
                full_loss = F.cross_entropy(output, target, reduction="none")
            else:
                raise ValueError(f"Unknown criterion {criterion}")

        preds.append(output)
        targets.append(target)
        losses.append(loss.item())
        full_losses.append(full_loss)

        if debug:
            break

    test_loss = torch.tensor(losses)
    full_losses = torch.cat(full_losses, dim=0)
    if args is not None:
        test_loss = utils.gather_tensor(test_loss, args)
    test_loss = test_loss.mean().item()
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    if args is not None:
        preds = utils.gather_tensor(preds, args)
        targets = utils.gather_tensor(targets, args)

    if task == "multilabel":
        test_acc = mean_average_precision(preds, targets) * 100
    elif task == "classification":
        test_acc = (preds.argmax(dim=-1) == targets).float().mean().item() * 100
    else:
        raise ValueError(f"Unknown criterion {criterion}")

    if not return_losses:
        return {"loss": test_loss, "acc": test_acc}
    else:
        return {"loss": test_loss, "acc": test_acc, "losses": full_losses}


def train_and_eval(
    model: torch.nn.Module,
    train_dataloader: Iterable,
    test_dataloader: Iterable,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad: float,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    warmup_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    warmup_epochs: int,
    device: torch.device,
    debug: bool = False,
    run=None,
    task: str = "multilabel",
    evaluate_every: int = 1,
):

    best_acc = 0.0
    best_model = None

    num_digits = len(str(epochs))
    metric = "mAP" if task == "multilabel" else "Acc"

    for epoch in range(epochs):
        train_stats = train(
            model=model,
            train_dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            clip_grad=clip_grad,
            warmup_scheduler=warmup_scheduler,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            device=device,
            debug=debug,
            task=task,
        )
        if epoch % evaluate_every == 0:
            test_stats = evaluate(
                model=model,
                test_dataloader=test_dataloader,
                criterion=criterion,
                device=device,
                debug=debug,
                task=task,
                args=None,
            )

        curr_lr = optimizer.param_groups[0]["lr"]

        log_string = f"| Epoch {str(epoch + 1).zfill(num_digits)}/{epochs} | Lr: {curr_lr:.4f} | Train Loss: {train_stats['loss']:.4f} | Train {metric}: {train_stats['acc']:.2f} |"
        if epoch % evaluate_every == 0:
            log_string += (
                f" Test Loss: {test_stats['loss']:.4f} | Test {metric}: {test_stats['acc']:.2f} |"
            )

        if "id_loss" in train_stats:
            log_string += f" Train ID Loss: {train_stats['id_loss']:.4f} | Train ID Acc: {train_stats['id_acc']:.2f} |"
        print(log_string)

        if run is not None:
            log_dict = {
                "train/lr": curr_lr,
                "train/loss": train_stats["loss"],
                f"train/{metric}": train_stats["acc"],
            }
            if epoch % evaluate_every == 0:
                log_dict["test/loss"] = test_stats["loss"]
                log_dict[f"test/{metric}"] = test_stats["acc"]
            if "id_loss" in train_stats:
                log_dict["train/id_loss"] = train_stats["id_loss"]
                log_dict["train/id_acc"] = train_stats["id_acc"]
            run.log(log_dict)

        if epoch >= warmup_epochs:
            scheduler.step()

        if test_stats["acc"] > best_acc:
            best_acc = test_stats["acc"]
            best_model = copy.deepcopy(model)

    print(f"| Best Test {metric}: {best_acc:.2f} | Last Test {metric}: {test_stats['acc']:.2f} |")
    return best_model, model
