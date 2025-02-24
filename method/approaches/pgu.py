##############################################################################
# Projected Gradient Unlearning
# Code adapted from https://github.com/hnanhtuan/projected_gradient_unlearning
##############################################################################

import torch
from torch.nn import functional as F
from method.engine import evaluate
from method.utils import freeze_norm_stats
from method.metrics import mean_average_precision


class HookRecoder:
    def __init__(self):
        self.hooks = {}

    def get_input(self, name):
        def hook_fn(module, input_, output):
            assert len(input_) == 1
            self.hooks[name] = {
                "type": type(module),
                "module": module,
                "input": input_[0],
            }

        return hook_fn


class ProjectedGradientUnlearning(torch.nn.Module):
    def __init__(self, gamma: float, epochs: int):
        super().__init__()
        self.gamma = gamma
        self.epochs = epochs

        self.svd = None
        self.retain_svd = None
        self.P = None
        self.ready = False

    def compute_CGS(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        forget_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        model.eval()
        # put stochastic layers in training mode
        for module in model.modules():
            name = module.__class__.__name__
            if name == "Dropout" or name == "DropPath" or name == "StochasticDepth":
                module.training = True

        self._compute_svd(model, train_dataloader, device)
        self._compute_retain_svd(model, forget_dataloader, device)
        self._compute_projection_matrices()
        self.ready = True

    def _reverse_ce_loss(self, logits, targets):
        softmax_logits = F.softmax(logits, dim=-1)
        loss = 0
        for i in range(logits.size(0)):
            logit = softmax_logits[i, targets[i]]
            loss += -torch.log(1 - logit + 0.05)

        return loss / logits.size(0)

    def _reverse_bce_loss(self, logits, targets):
        sigmoid_logits = F.sigmoid(logits)
        loss = 0
        for i in range(logits.size(0)):
            loss += -torch.log(1 - sigmoid_logits[i, targets[i]] + 0.05).mean()

        return loss / logits.size(0)

    @torch.no_grad()
    def _project_gradients(self, model, lr):
        for name, param in model.named_parameters():
            name = name.replace(".weight", "")
            if name not in self.P:
                param.grad.data.fill_(0)
                continue

            assert len(param.shape) > 1, "Only linear and conv layers are supported"
            reg_grad = param.grad.data.add(param.data, alpha=0)
            reg_grad = reg_grad - (reg_grad.view(reg_grad.size(0), -1) @ self.P[name]).view(
                param.size()
            )

            param.data -= lr * reg_grad

    def modify_weight(
        self,
        model: torch.nn.Module,
        forget_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        lr: float,
        device: torch.device,
        task: str,
    ) -> None:
        if not self.ready:
            raise ValueError("CGS is not computed yet")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        start_lr = lr
        end_lr = lr
        alpha = torch.exp(torch.log(torch.tensor(end_lr / start_lr)) / self.epochs)
        lr = start_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        loss2_w = 0.2

        num_digits = len(str(self.epochs))

        model.train()
        freeze_norm_stats(model)
        for epoch in range(self.epochs):
            losses = []
            accuracies = []
            for image, identity, target in forget_dataloader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    optimizer.zero_grad()
                    image = image.to(device=device, dtype=torch.bfloat16)
                    output = model(image)
                    if task == "classification":
                        loss1 = self._reverse_ce_loss(output, target)
                    else:
                        loss1 = self._reverse_bce_loss(output, target)
                    norm_output = F.softmax(output, dim=-1)
                    loss2 = -torch.sum(-norm_output * torch.log(norm_output + 1e-15), dim=-1).mean()
                    loss = loss1 + loss2_w * loss2

                    loss.backward()
                self._project_gradients(model, lr)

                if task == "classification":
                    acc = (output.argmax(dim=-1).cpu() == target).float().mean() * 100
                else:
                    acc = mean_average_precision(output.cpu(), target.cpu()) * 100

                losses.append(loss.item())
                accuracies.append(acc.item())

            # lr = lr * alpha
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            test_stats = evaluate(
                model=model,
                test_dataloader=test_dataloader,
                criterion=(
                    torch.nn.BCEWithLogitsLoss()
                    if task != "classification"
                    else torch.nn.CrossEntropyLoss()
                ),
                device=device,
                debug=False,
                task=task,
                args=None,
            )

            metric = "mAP" if task != "classification" else "Acc"
            print(
                f"| Epoch: {str(epoch+1).zfill(num_digits)}/{self.epochs} | Lr: {lr / alpha:.4f} | Loss: {sum(losses)/len(losses):.4f} | {metric}: {sum(accuracies)/len(accuracies):.2f} | Test Loss: {test_stats['loss']:.4f} | Test {metric}: {test_stats['acc']:.2f} |"
            )

    @torch.no_grad()
    def _compute_svd(
        self,
        model: torch.nn.Module,
        full_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """
        Computes the singular value decomposition for each layer on the entire dataset
        """
        hook_recorder = HookRecoder()
        hook_handles = []
        covar = {}
        # for each module, check if it is in the list of modules we want to record (linear, cov1d, cov2d)
        # if it is in the list, register a forward hook to record the input to the model
        # and init the covariance matrix for that layer to 0
        for name, module in model.named_modules():
            if type(module) in (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d):
                hook_handles.append(module.register_forward_hook(hook_recorder.get_input(name)))
                covar[name] = 0

        for image, _, _ in full_dataloader:
            image = image.to(device)
            model(image)

            # retrieve the input to the model for each layer and computes the covariance matrix
            for layer, record in hook_recorder.hooks.items():
                if record["type"] == torch.nn.Linear:
                    if len(record["input"].size()) == 3:
                        operation = "bti,btj->ij"
                    else:
                        operation = "bi,bj->ij"
                    covar[layer] += torch.einsum(operation, record["input"], record["input"])

                elif record["type"] in (torch.nn.Conv1d, torch.nn.Conv2d):
                    ks = record["module"].kernel_size
                    padding = record["module"].padding
                    features = record["input"]
                    patch = F.unfold(
                        features, kernel_size=ks, dilation=1, padding=padding, stride=1
                    )
                    fea_dim = patch.size(1)
                    patch = patch.permute(0, 2, 1).reshape(-1, fea_dim)
                    covar[layer] += torch.einsum("bi,bj->ij", patch, patch)

        svd = {}
        # for each layer, compute the singular value decomposition of the covariance matrix
        for layer, cov in covar.items():
            U, S, _ = torch.svd(cov)
            svd[layer] = (U, torch.sqrt(S))

        self.svd = svd

        for handle in hook_handles:
            handle.remove()

    @torch.no_grad()
    def _compute_retain_svd(
        self,
        model: torch.nn.Module,
        forget_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """
        Given the forget set, retrieves the singular value decomposition for each layer of the retain dataset
        """
        hook_recorder = HookRecoder()
        hook_handles = []
        covar = {}
        retain_svd = {}
        for name, module in model.named_modules():
            if type(module) in (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d):
                hook_handles.append(module.register_forward_hook(hook_recorder.get_input(name)))
                covar[name] = 0

        for image, _, _ in forget_dataloader:
            image = image.to(device)
            model(image)

            for layer, record in hook_recorder.hooks.items():
                if record["type"] == torch.nn.Linear:
                    if len(record["input"].size()) == 3:
                        operation = "bti,btj->ij"
                    else:
                        operation = "bi,bj->ij"
                    covar[layer] += torch.einsum(operation, record["input"], record["input"])

                elif record["type"] in (torch.nn.Conv1d, torch.nn.Conv2d):
                    ks = record["module"].kernel_size
                    padding = record["module"].padding
                    features = record["input"]
                    patch = F.unfold(
                        features, kernel_size=ks, dilation=1, padding=padding, stride=1
                    )
                    fea_dim = patch.size(1)
                    patch = patch.permute(0, 2, 1).reshape(-1, fea_dim)
                    covar[layer] += torch.einsum("bi,bj->ij", patch, patch)

        for layer, cov in covar.items():
            U, S = self.svd[layer]
            M = (U @ torch.diag(S**2)) @ U.t()

            M1 = M - cov
            U1_, S1_sqrt_, _ = torch.svd(M1)
            retain_svd[layer] = (U1_, torch.sqrt(S1_sqrt_))

        self.retain_svd = retain_svd

        for handle in hook_handles:
            handle.remove()

    def _compute_projection_matrices(self):
        """
        Computes the projection matrices for each layer
        """
        P = {}
        for layer, (U, S) in self.retain_svd.items():
            if self.gamma == 1:
                continue

            # TLDR: Retain the top k dimensions of the projection matrix with highest variance
            # From the dimension with highest variance, compute the comulative sum of the variances
            # then count the number of dimensions required to satisfy the gamma threshold.
            # This gives us the rank of the projection matrix.
            k = torch.sum((torch.cumsum(S, dim=0) / torch.sum(S)) <= self.gamma).item() + 1
            M = U[:, :k]
            P[layer] = M @ M.t()

        self.P = P
