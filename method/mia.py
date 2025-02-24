import torch
import torchvision
from functools import partial
from torch.nn import functional as F
from typing import List, Tuple

from method import utils
from method.dataset.dataset_classes import IdentityUnlearningDataset
from method.metrics import compute_roc_auc


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds = []
    labels = []
    for image, identity, target in dataloader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = image.to(device=device, dtype=torch.bfloat16)
            image_flipped = torchvision.transforms.RandomHorizontalFlip(p=1)(image)
            pred = model(image)
            pred_flipped = model(image_flipped)
        preds.append(torch.stack([pred, pred_flipped], dim=1).cpu().float())
        labels.append(target.cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    return preds, labels


@torch.no_grad()
def extract_scores_single(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    debug: bool,
):
    preds, labels = utils.predict(model, dataloader, device, debug)
    preds = F.sigmoid(preds)
    scores = torch.zeros(preds.size(0), device=device)
    for i in range(len(preds)):
        scores[i] = (torch.log(preds[i][labels[i] == 1]) - torch.log(1 - preds[i][labels[i] == 1] + 1e-30)).mean()
    return scores.cpu()


@torch.no_grad()
def extract_scores(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    debug: bool,
):
    preds, labels = predict(model, dataloader, device, debug)
    preds = preds - preds.amax(dim=(1, 2), keepdim=True)
    preds = F.sigmoid(preds)
    scores = torch.zeros_like(preds)
    for i in range(len(preds)):
        for j in range(preds.size(1)):
            for k in range(preds.size(2)):
                if labels[i, k] == 1:
                    scores[i, j, k] = torch.log(preds[i, j, k]) - torch.log(1 - preds[i, j, k] + 1e-30) 
                    # scores[i, j, k] = preds[i, j, k]
    return scores.cpu(), labels.cpu()


@torch.no_grad()
def estimate_gaussian_single(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    debug: bool,
):
    """Estimate the mean and variance of the model's output on the given dataloader."""
    scores = extract_scores(model, dataloader, criterion, device, debug)
    gaussian = torch.distributions.Normal(scores.mean(), scores.std())
    return gaussian


@torch.no_grad()
def estimate_gaussian(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    debug: bool,
) -> List[torch.distributions.Normal]:
    """Estimate a gaussian for each label in the model's output."""
    scores, labels = extract_scores(model, dataloader, criterion, device, debug)
    gaussians = []
    for k in range(scores.size(2)):
        score_k = []
        for i in range(scores.size(0)):
            if labels[i, k] == 1:
                score_k.append(scores[i, :, k])
        score_k = torch.cat(score_k)

        gaussian = torch.distributions.Normal(score_k.mean(), score_k.std())
        gaussians.append(gaussian)
    return gaussians


@torch.no_grad()
def estimate_likelihoods(scores: torch.Tensor, labels: torch.Tensor, gaussians: List[torch.distributions.Normal]):
    """Estimate the likelihood of the scores given the gaussians."""
    likelihoods = []
    for i in range(scores.size(0)):
        likelihood = 0
        for k in range(scores.size(2)):
            if labels[i, k] == 1:
                likelihood += 1 - gaussians[k].log_prob(scores[i, :, k]).mean()
        likelihood = likelihood / labels[i].sum()
        likelihoods.append(likelihood)
    return torch.stack(likelihoods)


@torch.no_grad()
def compute_mia(
    unlearned_model: torch.nn.Module,
    original_model: torch.nn.Module,
    target_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    debug: bool,
):
    """Compute Membership Inference Attack (MIA)"""
    # estimate the Gaussian distribution of the original model's output
    original_gaussian = estimate_gaussian(original_model, val_loader, criterion, device, debug)

    # estimate the Gaussian distribution of the unlearned model's output
    unlearned_gaussian = estimate_gaussian(unlearned_model, val_loader, criterion, device, debug)

    # compute scores of the forget data using the original model
    original_scores, original_targets = extract_scores(original_model, target_loader, criterion, device, debug)

    # compute scores of the forget data using the unlearned model
    unlearned_scores, unlearned_targets = extract_scores(unlearned_model, target_loader, criterion, device, debug)

    # compute the membership likelihood for the original model
    original_likelihood = estimate_likelihoods(original_scores, original_targets, original_gaussian)
    print(original_likelihood.mean(), sum(g.mean for g in original_gaussian) / len(original_gaussian))

    # compute the membership likelihood for the unlearned model
    unlearned_likelihood = estimate_likelihoods(unlearned_scores, unlearned_targets, unlearned_gaussian)
    print(unlearned_likelihood.mean(), sum(g.mean for g in unlearned_gaussian) / len(unlearned_gaussian))

    # compute the MIA score
    mia_score = (1 + unlearned_likelihood - original_likelihood) / 2
    return mia_score
    # likelihoods = torch.cat([original_likelihood, unlearned_likelihood])
    # membership = torch.cat([torch.zeros_like(original_likelihood), torch.ones_like(unlearned_likelihood)])

    # return likelihoods, membership


def compute_mia_auc_roc(
    unlearned_model: torch.nn.Module,
    original_model: torch.nn.Module,
    datasets: IdentityUnlearningDataset,
    criterion: torch.nn.Module,
    device: torch.device,
    args,
):

    retain_data = datasets.get_unlearning_data()["retain"]
    forget_data = datasets.get_unlearning_data()["forget"]
    val_data = datasets.get_val_data()
    test_data = datasets.get_test_data()

    partial_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    retain_dataloader = partial_loader(retain_data)
    forget_dataloader = partial_loader(forget_data)
    val_dataloader = partial_loader(val_data)
    test_dataloader = partial_loader(test_data)

    forget_mia_score = compute_mia(
        unlearned_model,
        original_model,
        forget_dataloader,
        test_dataloader,
        criterion,
        device,
        args.debug,
    )
    print(forget_mia_score.mean())
    val_mia_score = compute_mia(
        unlearned_model,
        original_model,
        val_dataloader,
        test_dataloader,
        criterion,
        device,
        args.debug,
    )
    print(val_mia_score.mean())
    mia_score = torch.cat([forget_mia_score, val_mia_score])

    forget_labels = torch.ones_like(forget_mia_score)
    val_labels = torch.zeros_like(val_mia_score)
    mia_labels = torch.cat([forget_labels, val_labels])

    # mia_score, mia_labels = forget_mia_score
    roc, auc = compute_roc_auc(mia_score, mia_labels)
    return roc, auc
