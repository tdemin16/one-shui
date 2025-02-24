import copy
import math
import torch
import warnings
from functools import partial
from tqdm import tqdm
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import List, Tuple, Callable

from method import utils
from method.dataset.dataset_classes import IdentityUnlearningDataset
from method.engine import evaluate
from method.metrics import mean_average_precision


class MetaLoss(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 0,
        num_identities: int = 0,
        identity_embed_size: int = 64,
        prob_dropout: float = 0.5,
        use_ids: bool = False,
    ):
        super().__init__()
        if use_ids:
            self.embed = torch.nn.Embedding(num_identities, identity_embed_size)

        self.input_block = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(prob_dropout),
        )
        hidden_blocks = []
        for i in range(num_layers):
            hidden_blocks.append(
                torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Dropout(prob_dropout),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Dropout(prob_dropout),
                )
            )
        self.hidden_blocks = torch.nn.ModuleList(hidden_blocks)
        self.output_block = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Softplus(),
        )

    def forward(self, x, identity, reduce: bool = True):
        if hasattr(self, "embed"):
            identity = self.embed(identity)
            x = torch.cat([x, identity], dim=1)
        # else:
        #     x = torch.cat([x, identity.unsqueeze(1)], dim=1)
        x = self.input_block(x)
        for block in self.hidden_blocks:
            x = x + block(x)
        x = self.output_block(x)
        if reduce:
            x = torch.mean(x)
        return x


class LossLearning:
    def __init__(
        self,
        model: torch.nn.Module,
        in_features: int,
        task_criterion: torch.nn.Module,
        identities: List[int],
        epochs: int,
        use_retain: bool,
        forget_epochs: int,
        forget_loss: str,
        use_accs: bool,
        debug: bool,
        run,
        args,
        task: str,
    ):
        self.model = model
        self.task_criterion = task_criterion
        self.epochs = epochs
        self.use_retain = use_retain
        self.forget_epochs = forget_epochs
        self.forget_loss = forget_loss
        self.use_accs = use_accs
        self.debug = debug
        self.run = run
        self.task = task
        self.identity_map = {identity: i for i, identity in enumerate(identities)}

        size = in_features
        if args.use_feats:
            size += 768
        if args.use_ids:
            size += 64
        if args.use_targets:
            size += in_features

        self.meta_forget_loss_fn = MetaLoss(
            input_size=size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_identities=len(identities),
            prob_dropout=args.prob_dropout,
            use_ids=args.use_ids,
        ).to(args.device)
        self.meta_retain_loss_fn = MetaLoss(
            input_size=size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_identities=len(identities),
            prob_dropout=args.prob_dropout,
            use_ids=args.use_ids,
        ).to(args.device)

    def fix_nn(self, model, theta):
        def k_param_fn(tmp_model, name=None):
            if len(tmp_model._modules) != 0:
                for k, v in tmp_model._modules.items():
                    if name is None:
                        k_param_fn(v, name=str(k))
                    else:
                        k_param_fn(v, name=str(name + "." + k))
            else:
                for k, v in tmp_model._parameters.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    tmp_model._parameters[k] = theta[str(name + "." + k)]

        k_param_fn(model)
        return model

    def generate_unlearned_model(self, model, grads, args):
        # create a state dict with updated model weights
        unlearned_weights = {}
        num_grad = 0
        for k, v in model.state_dict().items():
            if "running_mean" in k or "running_var" in k:
                unlearned_weights[k] = v
            else:
                unlearned_weights[k] = v - args.meta_lr * grads[num_grad]
                num_grad += 1
        # create a new model with the updated weights
        unlearned_model = copy.deepcopy(model)
        unlearned_model = unlearned_model.to(args.device)
        unlearned_model = self.fix_nn(unlearned_model, unlearned_weights)
        unlearned_model.eval()
        return unlearned_model

    def forget_loss_fn(self, f_logits, f_target, f_acc, v_logits, v_target, v_acc, **kwargs):
        """
        Forces forget and validation losses to be as close as possible
        """
        if self.task == "classification":
            forget_bce = F.cross_entropy(f_logits, f_target.long())
            val_bce = F.cross_entropy(v_logits, v_target.long())
        else:
            forget_bce = F.binary_cross_entropy_with_logits(f_logits, f_target.float())
            val_bce = F.binary_cross_entropy_with_logits(v_logits, v_target.float())

        if self.use_accs:
            forget_bce = forget_bce * (1 - f_acc)
            val_bce = val_bce * (1 - v_acc)

        if self.forget_loss == "l1":
            forget_loss = F.l1_loss(forget_bce, val_bce.detach())
        elif self.forget_loss == "l2":
            forget_loss = F.mse_loss(forget_bce, val_bce.detach())
        elif self.forget_loss == "smooth_l1":
            forget_loss = F.smooth_l1_loss(forget_bce, val_bce.detach())
        return forget_loss

    def retain_loss_fn(self, r_logits, r_target, **kwargs):
        """
        Preserves the model's performance on the retained data
        """
        retain_loss = F.binary_cross_entropy_with_logits(r_logits, r_target.float())
        return retain_loss

    def step(
        self,
        model: torch.nn.Module,
        meta_loss_fn: torch.nn.Module,
        support_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        forget_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        aux_loss_fn: Callable,
        args,
    ):
        """
        Generic meta learning loop
        """
        model.eval()
        model.zero_grad()

        grads = None
        meta_losses = []
        sizes = 0
        # "inner" loop
        for image, identity, target in support_loader:
            image = image.to(args.device)
            target = target.to(args.device)
            feats = model.forward_features(image)
            logits = F.sigmoid(model.forward_head(feats))

            # compute meta loss function
            if args.use_feats:
                logits = torch.cat([logits, feats[:, 0]], dim=1)
            if args.use_targets:
                if args.task == "classification":
                    target = F.one_hot(target, num_classes=args.num_classes)
                logits = torch.cat([logits, target], dim=1)
            mapped_identity = torch.tensor([self.identity_map[i.item()] for i in identity]).to(
                args.device
            )
            meta_loss = meta_loss_fn(logits, mapped_identity) * image.size(0)
            meta_losses.append(meta_loss.item())
            sizes += image.size(0)

            # compute gradients wrt model parameters (create_graph=True for differentiable gradients)
            grad = torch.autograd.grad(meta_loss, model.parameters(), create_graph=True)

            # sum up gradients
            if grads is None:
                grads = grad
            else:
                grads = [g + grad for g, grad in zip(grads, grad)]

            # clean model grad
            model.zero_grad()

            if args.debug:
                break

        # normalize gradients by loader size (makes sure every update has the same magnitude)
        grads = [g / sizes for g in grads]

        # generate an unlearned model using the computed gradients
        unlearned_model = self.generate_unlearned_model(model, grads, args)

        # "outer" loop
        aux_losses = 0
        sizes = 0
        # iterate over the forget loader (default number of batches per step)
        # iterate over the val loader if forget step, otherwise iterate over the retain loader
        iter_data = (
            iter(val_loader) if aux_loss_fn.__name__ == "forget_loss_fn" else iter(retain_loader)
        )
        iter_retain = None
        if aux_loss_fn.__name__ == "forget_loss_fn" and args.loss_type == "scrub":
            iter_retain = iter(retain_loader)
        for f_image, f_identity, f_target in forget_loader:
            g_image, g_identity, g_target = next(iter_data)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                f_image = f_image.to(device=args.device, dtype=torch.bfloat16)
                f_target = f_target.to(device=args.device, dtype=torch.bfloat16)
                g_image = g_image.to(device=args.device, dtype=torch.bfloat16)
                g_target = g_target.to(device=args.device, dtype=torch.bfloat16)

                # if forget step
                if aux_loss_fn.__name__ == "forget_loss_fn":
                    # original and unlearned model performance on the validation data
                    with torch.no_grad():
                        original_v = model(g_image)

                    # we need gradient if we want to align unlearned 
                    # and original models on the validation set
                    if args.loss_type == "rev":
                        unlearned_v = unlearned_model(g_image)
                    else:
                        with torch.no_grad():
                            unlearned_v = unlearned_model(g_image)

                    # unlearned model performance on the forget data
                    unlearned_f = unlearned_model(f_image)

                    # accs
                    if args.task == "classification":
                        original_v_acc = (original_v.argmax(dim=-1) == g_target).float().mean()
                        unlearned_v_acc = (unlearned_v.argmax(dim=-1) == g_target).float().mean()
                        unlearned_f_acc = (unlearned_f.argmax(dim=-1) == f_target).float().mean()
                    else:
                        original_v_acc = mean_average_precision(original_v, g_target.long())
                        unlearned_v_acc = mean_average_precision(unlearned_v, g_target.long())
                        unlearned_f_acc = mean_average_precision(unlearned_f, f_target.long())

                    # compute alignment between forget loss and validation losses
                    aux_loss = 0
                    if args.loss_type == "full" or args.loss_type == "original":
                        aux_loss += aux_loss_fn(
                            f_logits=unlearned_f,
                            f_target=f_target,
                            f_acc=unlearned_f_acc,
                            v_logits=original_v,
                            v_target=g_target,
                            v_acc=original_v_acc,
                        )
                    if (
                        args.loss_type == "full"
                        or args.loss_type == "unlearned"
                        or args.loss_type == "rev"
                    ):
                        aux_loss += aux_loss_fn(
                            f_logits=unlearned_f,
                            f_target=f_target,
                            f_acc=unlearned_f_acc,
                            v_logits=unlearned_v,
                            v_target=g_target,
                            v_acc=unlearned_v_acc,
                        )
                    if args.loss_type == "rev":
                        aux_loss += aux_loss_fn(
                            f_logits=unlearned_v,
                            f_target=g_target,
                            f_acc=unlearned_v_acc,
                            v_logits=original_v,
                            v_target=g_target,
                            v_acc=original_v_acc,
                        )

                    if args.loss_type == "scrub":
                        r_image, r_identity, r_target = next(iter_retain)
                        r_image = r_image.to(device=args.device, dtype=torch.bfloat16)
                        r_target = r_target.to(device=args.device, dtype=torch.bfloat16)
                        unlearned_r = unlearned_model(r_image)
                        with torch.no_grad():
                            original_r = model(r_image)
                            original_f = model(f_image)
                        aux_loss = 0.99 * F.binary_cross_entropy_with_logits(unlearned_r, r_target)
                        aux_loss += (
                            0.001
                            * F.kl_div(
                                F.log_softmax(unlearned_r / 4, dim=-1),
                                F.softmax(original_r / 4, dim=-1),
                            )
                            * 4**2
                        )
                        aux_loss -= (
                            F.kl_div(
                                F.log_softmax(unlearned_f / 4, dim=-1),
                                F.softmax(original_f / 4, dim=-1),
                            )
                            * 4**2
                        )

                # if retain step
                else:
                    # unlearned model performance on the retain data
                    unlearned_r = unlearned_model(g_image)

                    # compute task loss on the retain data
                    aux_loss = aux_loss_fn(
                        r_logits=unlearned_r,
                        r_target=g_target,
                    )

                # accumulate loss over all batches
                aux_losses += aux_loss * f_image.size(0)
                sizes += f_image.size(0)

            if args.debug:
                break

        # decouple loss magnitude from dataloader len
        aux_losses /= sizes

        # perfom meta update
        aux_losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        detached_grads = [grad.detach().clone() for grad in grads]

        return {
            "aux_loss": aux_losses.item(),
            "grads": detached_grads,
            "meta_loss": sum(meta_losses) / len(meta_losses),
            "grads_magnitude": sum([grad.norm().item() for grad in detached_grads]),
        }

    @sdpa_kernel(SDPBackend.MATH)
    def train(
        self,
        datasets: IdentityUnlearningDataset,
        optimizer_forget: torch.optim.Optimizer,
        optimizer_retain: torch.optim.Optimizer,
        args,
    ):
        self.meta_forget_loss_fn.train()
        self.meta_retain_loss_fn.train()
        self.model.eval()

        if args.robustness:
            if args.dataset == "celebahq":
                sim_num_ids = 20 if args.num_identities == 50 else 50
            else:
                sim_num_ids = 5 if args.num_identities == 10 else 10
        else:
            sim_num_ids = args.num_identities_simulation

        partial_loader = partial(
            torch.utils.data.DataLoader,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )

        meta_losses = []
        forget_losses = []
        retain_losses = []
        bar = tqdm(
            total=len(datasets.get_simulated_unlearning_data(num_ids=sim_num_ids)),
            leave=False,
            dynamic_ncols=True,
        )
        # iterate over all multiple unlearning datasets
        for train_dataset, test_dataset in datasets.get_simulated_unlearning_data(
            num_ids=sim_num_ids
        ):
            support_loader = partial_loader(test_dataset["support"])
            retain_loader = (
                partial_loader(train_dataset["retain"])
                if args.use_train_aug
                else partial_loader(test_dataset["retain"])
            )
            forget_loader = partial_loader(test_dataset["forget"])
            val_loader = partial_loader(datasets.get_val_data())

            unlearned_model = copy.deepcopy(self.model)
            for i in range(1):
                # perform meta forget step
                # - use support set to compute the loss and create an update of the model
                # - use forget set and validation set to evaluate the computed model
                # - update the meta_forget_loss_fn
                forget_step = self.step(
                    model=unlearned_model,
                    meta_loss_fn=self.meta_forget_loss_fn,
                    support_loader=support_loader,
                    retain_loader=retain_loader,
                    forget_loader=forget_loader,
                    val_loader=val_loader,
                    optimizer=optimizer_forget,
                    aux_loss_fn=self.forget_loss_fn,
                    args=args,
                )
                forget_losses.append(forget_step["aux_loss"])
                meta_losses.append(forget_step["meta_loss"])

                # use the gradients computed in the forget step to generate an unlearned model
                unlearned_model = self.generate_unlearned_model(
                    unlearned_model, forget_step["grads"], args
                )
                utils.set_params(unlearned_model, requires_grad=True)

            # perform meta retain step
            # - use support set to compute the loss and create an update of the unleared model
            # - use retain set to evaluate the computed model
            # - update the meta_retain_loss_fn
            if self.use_retain:
                retain_step = self.step(
                    model=unlearned_model,
                    meta_loss_fn=self.meta_retain_loss_fn,
                    support_loader=support_loader,
                    retain_loader=retain_loader,
                    forget_loader=forget_loader,
                    val_loader=val_loader,
                    optimizer=optimizer_retain,
                    aux_loss_fn=self.retain_loss_fn,
                    args=args,
                )
                retain_losses.append(retain_step["aux_loss"])
                meta_losses.append(retain_step["meta_loss"])

            # set pbar description
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
            with warnings.catch_warnings(action="ignore"):
                torch.cuda.reset_max_memory_allocated()
            pbar_desc = f"| Meta Loss: {sum(meta_losses) / len(meta_losses):.4f} | Grad Magnitudes: {forget_step['grads_magnitude']:.4f} "
            if self.use_retain:
                pbar_desc += f"- {retain_step['grads_magnitude']:.4f} "
            pbar_desc += f"| Forget Aux Loss: {sum(forget_losses) / len(forget_losses):.4f} "
            if self.use_retain:
                pbar_desc += f"| Retain Aux Loss: {sum(retain_losses) / len(retain_losses):.4f} "
            pbar_desc += f"| GPU Memory: {gpu_memory:.2f} GB |"
            bar.set_description(pbar_desc)
            bar.update(1)

            if args.debug:
                break
        bar.close()

        return {
            "meta_loss": sum(meta_losses) / len(meta_losses),
            "retain_loss": sum(retain_losses) / len(retain_losses) if self.use_retain else 0,
            "forget_loss": sum(forget_losses) / len(forget_losses),
        }

    def test(self, datasets: IdentityUnlearningDataset, args):
        model = copy.deepcopy(self.model)
        model = model.to(args.device)
        model.eval()

        unlearning_dataset = datasets.get_unlearning_data()
        val_data = datasets.get_val_data()

        partial_loader = partial(
            torch.utils.data.DataLoader,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        support_loader = partial_loader(unlearning_dataset["support"], shuffle=True)
        retain_loader = partial_loader(unlearning_dataset["retain"], shuffle=False)
        forget_loader = partial_loader(unlearning_dataset["forget"], shuffle=False)
        val_loader = partial_loader(val_data, shuffle=False)

        unlearned_model = self.unlearn_model(support_loader, args)
        unlearned_model.eval()
        utils.set_params(unlearned_model, requires_grad=False)

        partial_evaluate = partial(
            evaluate,
            model=unlearned_model,
            criterion=self.task_criterion,
            device=args.device,
            debug=self.debug,
            task=self.task,
            args=args,
        )
        retain_stats = partial_evaluate(test_dataloader=retain_loader)
        forget_stats = partial_evaluate(test_dataloader=forget_loader)
        val_stats = partial_evaluate(test_dataloader=val_loader)

        return {"retain": retain_stats, "forget": forget_stats, "test": val_stats}

    def train_loss(self, datasets: IdentityUnlearningDataset, args):
        num_digits = len(str(self.epochs))

        optimizer_forget = torch.optim.AdamW(
            self.meta_forget_loss_fn.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=True,
        )
        optimizer_retain = torch.optim.AdamW(
            self.meta_retain_loss_fn.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=True,
        )
        scheduler_forget = utils.get_scheduler(optimizer_forget, args)
        scheduler_retain = utils.get_scheduler(optimizer_retain, args)

        for epoch in range(self.epochs):
            train_stats = self.train(
                datasets=datasets,
                optimizer_forget=optimizer_forget,
                optimizer_retain=optimizer_retain,
                args=args,
            )

            test_stats = self.test(
                datasets=datasets,
                args=args,
            )

            print(
                f"| Epoch {str(epoch + 1).zfill(num_digits)}/{self.epochs} | Meta Loss: {train_stats['meta_loss']:.4f} |"
            )
            print_str = f"| Forget Aux Loss: {train_stats['forget_loss']:.4f} |"
            if self.use_retain:
                print_str += f" Retain Aux Loss: {train_stats['retain_loss']:.4f} |"
            print(print_str)
            print(
                f"| Retain Loss: {test_stats['retain']['loss']:.4f} | Retain mAP: {test_stats['retain']['acc']:.4f} | Forget Loss: {test_stats['forget']['loss']:.4f} | Forget mAP: {test_stats['forget']['acc']:.4f} | Test Loss: {test_stats['test']['loss']:.4f} | Test mAP: {test_stats['test']['acc']:.4f} |"
            )
            print()

            if self.run is not None:
                log_dict = {
                    "epoch": epoch,
                    "meta_loss": train_stats["meta_loss"],
                    "aux_loss/forget": train_stats["forget_loss"],
                    "aux_loss/retain": train_stats["retain_loss"],
                    "retain_mAP": test_stats["retain"]["acc"],
                    "retain_loss": test_stats["retain"]["loss"],
                    "forget_mAP": test_stats["forget"]["acc"],
                    "forget_loss": test_stats["forget"]["loss"],
                    "test_mAP": test_stats["test"]["acc"],
                    "test_loss": test_stats["test"]["loss"],
                }
                self.run.log(log_dict)

            scheduler_forget.step()
            scheduler_retain.step()

    def _unlearn_step(self, model, dataloader, loss_fn, args):
        grads = None
        sizes = 0
        for image, identity, target in dataloader:
            image = image.to(args.device)
            target = target.to(args.device)
            feats = model.forward_features(image)
            logits = F.sigmoid(model.forward_head(feats))

            if args.use_feats:
                logits = torch.cat([logits, feats[:, 0]], dim=1)
            if args.use_targets:
                if args.task == "classification":
                    target = F.one_hot(target, num_classes=args.num_classes)
                logits = torch.cat([logits, target], dim=1)
            mapped_identity = torch.tensor([self.identity_map[i.item()] for i in identity]).to(
                args.device
            )
            meta_loss = loss_fn(logits, mapped_identity) * image.size(0)
            sizes += image.size(0)

            grad = torch.autograd.grad(meta_loss, model.parameters())
            if grads is None:
                grads = grad
            else:
                grads = [g + grad for g, grad in zip(grads, grad)]
            model.zero_grad()

            if args.debug:
                break

        grads = [g / sizes for g in grads]
        unlearned_model = self.generate_unlearned_model(model, grads, args)
        utils.set_params(unlearned_model, requires_grad=True)
        model.zero_grad()
        return unlearned_model

    def unlearn_model(self, dataloader, args):
        self.meta_forget_loss_fn.eval()
        self.meta_retain_loss_fn.eval()
        utils.set_params(self.meta_forget_loss_fn, requires_grad=False)
        utils.set_params(self.meta_retain_loss_fn, requires_grad=False)

        unlearned_model = copy.deepcopy(self.model)
        for i in range(self.forget_epochs):
            unlearned_model = self._unlearn_step(
                unlearned_model, dataloader, self.meta_forget_loss_fn, args
            )

        if self.use_retain:
            unlearned_model = self._unlearn_step(
                unlearned_model, dataloader, self.meta_retain_loss_fn, args
            )

        self.model.zero_grad()
        utils.set_params(self.meta_forget_loss_fn, requires_grad=True)
        utils.set_params(self.meta_retain_loss_fn, requires_grad=True)

        return unlearned_model
