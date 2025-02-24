import copy
import torch
from functools import partial
from torch.nn import functional as F
from tqdm import tqdm

from method import utils
from method.engine import evaluate


class TeacherDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        forget_data: torch.utils.data.Dataset,
        retain_data: torch.utils.data.Dataset,
    ):
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.forget_len + self.retain_len

    def __getitem__(self, idx):
        if idx < self.forget_len:
            return *self.forget_data[idx], 1
        else:
            return *self.retain_data[idx - self.forget_len], 0


class BadTeacher:
    def __init__(
        self,
        model: torch.nn.Module,
        dumb_teacher: torch.nn.Module,
        debug: bool,
        run,
        args,
    ):
        self.model = model
        self.dumb_teacher = dumb_teacher
        self.debug = debug
        self.run = run
        self.epochs = args.epochs

    def loss(
        self,
        student_preds: torch.Tensor,
        splits: torch.Tensor,
        dumb_teacher_preds: torch.Tensor,
        smart_teacher_preds: torch.Tensor,
    ):
        dumb_teacher_preds = F.softmax(dumb_teacher_preds, dim=1)
        smart_teacher_preds = F.softmax(smart_teacher_preds, dim=1)
        splits = splits.unsqueeze(-1)
        teacher_preds = splits * dumb_teacher_preds + (1 - splits) * smart_teacher_preds
        student_preds = F.log_softmax(student_preds, dim=1)
        return F.kl_div(student_preds, teacher_preds, reduction="batchmean")

    def validate(self, model, retain_dataset, forget_dataset, val_dataset, args):
        partial_loader = partial(
            torch.utils.data.DataLoader,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.pin_memory,
        )
        retain_loader = partial_loader(retain_dataset)
        forget_loader = partial_loader(forget_dataset)
        val_loader = partial_loader(val_dataset)

        retain_stats = evaluate(
            model,
            retain_loader,
            utils.get_criterion(args.criterion),
            args.device,
            args.debug,
            args.task,
            args,
        )
        forget_stats = evaluate(
            model,
            forget_loader,
            utils.get_criterion(args.criterion),
            args.device,
            args.debug,
            args.task,
            args,
        )
        val_stats = evaluate(
            model,
            val_loader,
            utils.get_criterion(args.criterion),
            args.device,
            args.debug,
            args.task,
            args,
        )

        return retain_stats, forget_stats, val_stats

    def unlearn_model(
        self,
        retain_dataset: torch.utils.data.Dataset,
        forget_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        args,
        validate: bool=True
    ):
        unlearning_data = TeacherDataset(forget_dataset, retain_dataset)
        unlearn_loader = torch.utils.data.DataLoader(
            unlearning_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        unlearning_model = copy.deepcopy(self.model)
        unlearning_model.train()
        self.model.eval()
        self.dumb_teacher.eval()

        optimizer = utils.get_optimizer(unlearning_model, args)
        scheduler = utils.get_scheduler(optimizer, args)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.epochs):
            losses = []
            for image, identity, target, split in unlearn_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image = image.to(device=args.device, dtype=torch.bfloat16)
                    split = split.to(device=args.device, dtype=torch.bfloat16)

                    pred = unlearning_model(image)
                    with torch.no_grad():
                        dumb_pred = self.dumb_teacher(image)
                        smart_pred = self.model(image)

                    loss = self.loss(pred, split, dumb_pred, smart_pred)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                losses.append(loss.item())
                if self.debug:
                    break

            if self.debug:
                break

            scheduler.step()

            if validate:
                retain_stats, forget_stats, val_stats = self.validate(
                    unlearning_model, retain_dataset, forget_dataset, val_dataset, args
                )
                print(
                    f"| Epoch: {epoch} | Retain Loss: {retain_stats['loss']:.4f} | Retain Acc: {retain_stats['acc']:.2f} | Forget Loss: {forget_stats['loss']:.4f} | Forget Acc: {forget_stats['acc']:.2f} | Val Loss: {val_stats['loss']:.4f} | Val Acc: {val_stats['acc']:.2f} |"
                )

        return unlearning_model
