'''
Code adapted from RMIA original codebase
'''


import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
from functools import partial
from sklearn.metrics import roc_auc_score, roc_curve

from method import utils
from method.configs import parse_args
from method.dataset import get_datasets
from method.dataset.dataset_classes import IdentityUnlearningDataset
from method.models import get_model


AUG = 8


def convert_signal(logits, labels):
    if len(labels.size()) == 2:
        log_p = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none").mean(
            dim=-1
        )
    else:
        log_p = F.cross_entropy(logits, labels, reduction="none")
    return log_p


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def get_probs(
    target_model: torch.nn.Module,
    reference_model: torch.utils.data.DataLoader,
    dataloader: torch.utils.data.DataLoader,
    args,
):
    target_model.eval()
    reference_model.eval()

    p_x_theta = []  # prob of x in target model
    p_x = []  # prob of x avg over out models (1 in our case)
    for image, identities, target in dataloader:
        image = image.to(device=args.device, dtype=torch.bfloat16)
        target = target.to(args.device)

        p_x_theta.append(convert_signal(target_model(image), target))
        p_x.append(convert_signal(reference_model(image), target))

    p_x_theta = torch.cat(p_x_theta, dim=0)
    p_x = torch.cat(p_x, dim=0)

    return p_x_theta, p_x


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def majority_voting_tensor(
    tensor, axis
):  # compute majority voting for a bool tensor along a certain axis
    return torch.mode(torch.stack(tensor), axis).values * 1.0


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def compute_mia_metric(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    datasets: IdentityUnlearningDataset,
    args,
):
    train_data = datasets.get_train_data(train=True)
    forget_data = datasets.get_unlearning_data(train=True)["forget"]
    val_data = datasets.get_val_data(train=True)

    generic_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_memory,
    )
    train_loader = generic_loader(train_data)
    forget_loader = generic_loader(forget_data)
    val_loader = generic_loader(val_data)

    all_z_in, all_z_avg = [], []
    all_forget_in, all_forget_avg = [], []
    all_val_in, all_val_avg = [], []
    for i in range(AUG):
        train_in, train_avg = get_probs(target_model, reference_model, train_loader, args)
        val_in, val_avg = get_probs(target_model, reference_model, val_loader, args)
        forget_in, forget_avg = get_probs(target_model, reference_model, forget_loader, args)

        z_in = torch.cat((train_in, val_in), dim=0)
        z_avg = torch.cat((train_avg, val_avg), dim=0)

        if args.dataset == "celeba":
            z_indices = torch.randperm(z_in.size(0))[: min(z_in.size(0), 30_000)]
            z_in = z_in[z_indices]
            z_avg = z_avg[z_indices]

            v_indices = torch.randperm(val_in.size(0))[: min(val_in.size(0), 7_000)]
            val_in = val_in[v_indices]
            val_avg = val_avg[v_indices]

        all_z_in.append(z_in)
        all_z_avg.append(z_avg)
        all_forget_in.append(forget_in)
        all_forget_avg.append(forget_avg)
        all_val_in.append(val_in)
        all_val_avg.append(val_avg)

    # [AUG, D]
    z_in = torch.stack(all_z_in, dim=0)
    z_avg = torch.stack(all_z_avg, dim=0)
    forget_in = torch.stack(all_forget_in, dim=0)
    forget_avg = torch.stack(all_forget_avg, dim=0)
    val_in = torch.stack(all_val_in, dim=0)
    val_avg = torch.stack(all_val_avg, dim=0)

    best_tpr_at_fpr = 0
    best_auc = 0
    best_fpr = 0
    best_tpr = 0
    gammas = [1, 2, 4, 8]
    as_ = [0.2, 0.3, 0.5, 1]
    for gamma, a in itertools.product(*[gammas, as_]):
        forget_scores = []
        val_scores = []
        for i in range(AUG):
            z_ratio = 1 / (z_in[i] / z_avg[i]) / ((1 + a) / 2 * z_avg[i] + (1 - a) / 2)
            forget_ratio = (
                forget_in[i] / forget_avg[i] / ((1 + a) / 2 * forget_avg[i] + (1 - a) / 2)
            )
            val_ratio = val_in[i] / val_avg[i] / ((1 + a) / 2 * val_avg[i] + (1 - a) / 2)

            forget_lr = torch.outer(forget_ratio, z_ratio)
            val_lr = torch.outer(val_ratio, z_ratio)

            forget_scores.append(forget_lr > gamma)
            val_scores.append(val_lr > gamma)

        forget_score = -(majority_voting_tensor(forget_scores, axis=0).mean(dim=1))
        val_score = -(majority_voting_tensor(val_scores, axis=0).mean(dim=1))

        preds = torch.cat((forget_score, val_score), dim=0).cpu().numpy()
        targets = (
            torch.cat((torch.ones_like(forget_score), torch.zeros_like(val_score)), dim=0)
            .cpu()
            .numpy()
        )

        auc = roc_auc_score(targets, preds)
        fpr, tpr, thr = roc_curve(targets, preds)

        tpr_at_fpr = np.interp(0.0001, fpr, tpr)
        if tpr_at_fpr > best_tpr_at_fpr:
            best_tpr_at_fpr = tpr_at_fpr
            best_auc = auc
            best_fpr = fpr
            best_tpr = tpr

    return best_auc, best_fpr, best_tpr, best_tpr_at_fpr


def load_checkpoints(checkpoint_dir, dataset, model, method, num_ids, seed):
    splits = None
    state_dict = None
    checkpoint_path = os.path.join(
        checkpoint_dir,
        dataset,
        args.model,
        f"{method}_{num_ids}_{seed}.pth",
    )
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        splits = checkpoint["splits"]
        state_dict = checkpoint["state_dict"]
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    model = get_model(
        model_name=args.model,
        num_classes=40 if args.dataset != "mufac" else 8,
        size=224,
        pretrained=args.pretrained,
    )

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    return model, splits


def main(args):
    utils.init_distributed_mode(args)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    utils.seed_everything(args.seed)
    plt.style.use(["science"])

    _, splits = load_checkpoints(
        "checkpoints", args.dataset, args.model, "pretrain", args.num_identities, 0
    )

    datasets = get_datasets(args, splits)

    METHODS = ["pretrain", "lipschitz", "pgu", "ssd", "meta_unlearn", "bad_teacher", "scrub"]
    for method in METHODS:
        target_model, _ = load_checkpoints(
            "checkpoints", args.dataset, args.model, method, args.num_identities, 0
        )

        reference_method = method if method != "pretrain" else "retrain"
        shadow_model_dir = "shadow_models" if method != "pretrain" else "checkpoints"
        reference_model, _ = load_checkpoints(
            shadow_model_dir, args.dataset, args.model, reference_method, args.num_identities, 1 if method != "pretrain" else 0
        )

        target_model = target_model.to(args.device)
        reference_model = reference_model.to(args.device)

        auc, fpr, tpr, tpr_at_fpr = compute_mia_metric(
            target_model, reference_model, datasets, args
        )
        print(f"{method}: AUC {auc * 100} - TPR@0.01%FPR {tpr_at_fpr * 100:.2f}%")

        plt.plot(fpr, tpr, label=f"{method}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="k", label="Random chance")
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)

    # Axis labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MIA attack success rate in log scale")

    # Display the legend
    plt.legend(loc="lower right")

    # Show the plot
    plt.grid()
    plt.savefig(f"roc_low_cost.pdf")


if __name__ == "__main__":
    args = parse_args()

    main(args)
