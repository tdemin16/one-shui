import copy
import torch
from functools import partial
from tqdm import tqdm

from method import utils
from method.metrics import mean_average_precision


class SCRUB:
    def __init__(
        self,
        model: torch.nn.Module,
        task_criterion: torch.nn.Module,
        debug: bool,
        run,
        args,
        task: str,
    ):
        self.model = model
        self.task_criterion = task_criterion
        self.debug = debug
        self.run = run
        self.alpha = args.alpha_scrub
        self.gamma = args.gamma_scrub
        self.T = args.temperature_scrub
        self.epochs = args.epochs
        self.forgetting_epochs = args.forgetting_epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.task = task

    def retain_loss(
        self,
        original_r: torch.Tensor,
        unlearned_r: torch.Tensor,
        task_labels: torch.Tensor,
    ) -> torch.Tensor:
        unlearned_r_norm = torch.log_softmax(unlearned_r / self.T, dim=1)
        original_r_norm = torch.softmax(original_r / self.T, dim=1)

        align_retain = (
            torch.nn.functional.kl_div(
                unlearned_r_norm, original_r_norm, reduction="sum"
            )
            * (self.T**2)
            / original_r.size(0)
        )

        if self.task == "classification":
            task_loss = self.task_criterion(unlearned_r, task_labels)
        else:
            task_loss = self.task_criterion(unlearned_r, task_labels.float())
        return self.alpha * align_retain + self.gamma * task_loss

    def foget_loss(
        self, original_f: torch.Tensor, unlearned_f: torch.Tensor
    ) -> torch.Tensor:
        unlearned_f_norm = torch.log_softmax(unlearned_f / self.T, dim=1)
        original_f_norm = torch.softmax(original_f / self.T, dim=1)

        align_forget = (
            torch.nn.functional.kl_div(
                unlearned_f_norm, original_f_norm, reduction="sum"
            )
            * (self.T**2)
            / original_f.size(0)
        )
        return -align_forget

    @torch.no_grad()
    def validate(self, model, retain_loader, forget_loader, val_loader, epoch, args):
        model.eval()
        retain_preds = []
        forget_preds = []
        val_preds = []
        retain_targets = []
        forget_targets = []
        val_targets = []

        for name, loader in zip(
            ["retain", "forget", "val"], [retain_loader, forget_loader, val_loader]
        ):
            for images, identities, targets in loader:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    images = images.to(device=args.device, dtype=torch.bfloat16)
                    targets = targets.to(device=args.device)
                    logits = model(images)
                if name == "retain":
                    retain_preds.append(logits)
                    retain_targets.append(targets)
                elif name == "forget":
                    forget_preds.append(logits)
                    forget_targets.append(targets)
                elif name == "val":
                    val_preds.append(logits)
                    val_targets.append(targets)
                if self.debug:
                    break

        retain_preds = torch.cat(retain_preds, dim=0)
        forget_preds = torch.cat(forget_preds, dim=0)
        val_preds = torch.cat(val_preds, dim=0)
        retain_targets = torch.cat(retain_targets, dim=0)
        forget_targets = torch.cat(forget_targets, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        
        if self.task == "classification":
            retain_loss = self.task_criterion(retain_preds, retain_targets)
            forget_loss = self.task_criterion(forget_preds, forget_targets)
            val_loss = self.task_criterion(val_preds, val_targets)

            # actually acc
            retain_mAP = (retain_preds.argmax(dim=-1) == retain_targets).float().mean() * 100
            forget_mAP = (forget_preds.argmax(dim=-1) == forget_targets).float().mean() * 100
            val_mAP = (val_preds.argmax(dim=-1) == val_targets).float().mean() * 100
        else:
            retain_loss = self.task_criterion(retain_preds, retain_targets.float())
            forget_loss = self.task_criterion(forget_preds, forget_targets.float())
            val_loss = self.task_criterion(val_preds, val_targets.float())

            retain_mAP = mean_average_precision(retain_preds, retain_targets) * 100
            forget_mAP = mean_average_precision(forget_preds, forget_targets) * 100
            val_mAP = mean_average_precision(val_preds, val_targets) * 100

        print(
            f" * Validation | Retain Loss: {retain_loss:.4f} | Retain mAP: {retain_mAP:.2f} | Forget Loss: {forget_loss:.4f} | Forget mAP: {forget_mAP:.2f} | Val Loss: {val_loss:.4f} | Val mAP: {val_mAP:.2f} |"
        )

    def unlearn_model(
        self,
        retain_loader: torch.utils.data.DataLoader,
        forget_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        args,
        validate: bool=True,
    ):
        model = copy.deepcopy(self.model)

        optimizer = utils.get_optimizer(model, args)
        scheduler = utils.get_scheduler(optimizer, args)

        scaler = torch.cuda.amp.GradScaler()
        pbar = partial(tqdm, leave=False, dynamic_ncols=True)
        for epoch in range(self.epochs):
            model.train()
            print_str = f" * Training | LR: {scheduler.get_last_lr()[0]:.6f} |"
            if epoch < self.forgetting_epochs:
                forget_losses = []
                for image, identity, target in pbar(
                    forget_loader,
                    desc=f"| Epoch {epoch+1}/{self.epochs} | Forgetting |",
                ):  
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        image = image.to(device=args.device, dtype=torch.bfloat16)
                        pred = model(image)
                        with torch.no_grad():
                            original_pred = self.model(image)

                        loss = self.foget_loss(original_pred, pred)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    forget_losses.append(loss.item())
                    if self.debug:
                        break
                
                print_str += f" Forget Loss: {sum(forget_losses)/len(forget_losses):.4f} |"

            retain_losses = []
            for image, identity, target in pbar(
                retain_loader, desc=f"| Epoch {epoch+1}/{self.epochs} | Retaining |"
            ):  
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    image = image.to(device=args.device, dtype=torch.bfloat16)
                    target = target.to(device=args.device)
                    pred = model(image)
                    with torch.no_grad():
                        original_pred = self.model(image)

                    loss = self.retain_loss(original_pred, pred, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                retain_losses.append(loss.item())
                if self.debug:
                    break
            
            print_str += f" Retain Loss: {sum(retain_losses)/len(retain_losses):.4f} |"
            if validate:
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(print_str)

                self.validate(model, retain_loader, forget_loader, val_loader, epoch, args)
                print()

            if self.debug:
                break

            scheduler.step()

        return model
