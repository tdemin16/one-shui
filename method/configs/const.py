class Const:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 8
        self.lr = 0.1
        self.clip_grad = 0.0
        self.momentum = 0.0
        self.weight_decay = 0.0
        self.optimizer = "sgd"
        self.amsgrad = False
        self.scheduler = "const"
        self.t_max = float("inf")
        self.step_size = 5
        self.milestones = [7]
        self.gamma = 0.2
        self.warmup_epochs = 0
        self.evaluate_every = 1
        self.meta_lr = 0.1
        self.lipschitz_weighting = 0.0
        self.forgetting_epochs = 0.0
        self.use_train_aug = True
        self.scale_size_forget = 1
        self.scale_size_retain = 1
        self.ssd_dampening_constant = 0.1
        self.ssd_selection_weighting = 0.1
        self.pgu_gamma = 0.9
        self.num_identities_simulation = 0


def get_const(tmp_args) -> Const:
    const = Const()

    # ? Model and dataset specific constants
    if tmp_args.method in ("pretrain", "retrain") and tmp_args.dataset == "celebahq" and "resnet" in tmp_args.model:
        const.epochs = 100
        const.scheduler = "multistep"
        const.milestones = [30, 60, 90]

    elif (
        tmp_args.method in ("pretrain", "retrain")
        and tmp_args.dataset in ("celebahq", "celeba", "mufac")
        and "vit" in tmp_args.model
    ):
        const.epochs = 30
        const.lr = 1e-3
        const.clip_grad = 0.0
        const.momentum = 0.9
        const.weight_decay = 1e-3 if tmp_args.dataset == "celebahq" else 1e-4
        const.scheduler = "cosine"
        const.warmup_epochs = 2

    # ? Method specific constants
    if tmp_args.method == "ssd":
        const.batch_size = 128
        if tmp_args.dataset == "celebahq":
            if tmp_args.num_identities <= 20:
                const.ssd_dampening_constant = 0.1
                const.ssd_selection_weighting = 50
            elif tmp_args.num_identities == 50:
                const.ssd_dampening_constant = 0.5
                const.ssd_selection_weighting = 10
            elif tmp_args.num_identities == 200:
                const.ssd_dampening_constant = 0.5
                const.ssd_selection_weighting = 10
        elif tmp_args.dataset == "celeba":
            if tmp_args.num_identities <= 20:
                const.ssd_dampening_constant = 2
                const.ssd_selection_weighting = 50
            elif tmp_args.num_identities == 50:
                const.ssd_dampening_constant = 2
                const.ssd_selection_weighting = 50
            elif tmp_args.num_identities == 200:
                const.ssd_dampening_constant = 2
                const.ssd_selection_weighting = 50
        elif tmp_args.dataset == "mufac":
            if tmp_args.num_identities == 5:
                const.ssd_dampening_constant = 0.1
                const.ssd_selection_weighting = 50
            if tmp_args.num_identities == 10:
                const.ssd_dampening_constant = 0.1
                const.ssd_selection_weighting = 1

    if tmp_args.method == "lipschitz":
        const.batch_size = 128
        const.lr = 0.01
        if "resnet" in tmp_args.model:
            const.lipschitz_weighting = 0.01
        elif "vit" in tmp_args.model:
            const.lipschitz_weighting = 30.0

    if tmp_args.method == "pgu":
        const.batch_size = 128
        const.epochs = 5
        const.lr = 0.1
        const.pgu_gamma = 0.8
        if tmp_args.dataset == "celebahq":
            if tmp_args.num_identities <= 20:
                const.lr = 0.1
                const.epochs = 5
                const.pgu_gamma = 0.9
                const.use_train_aug = False
            elif tmp_args.num_identities == 50:
                const.lr = 0.1
                const.epochs = 5
                const.pgu_gamma = 0.9
                const.use_train_aug = False
            elif tmp_args.num_identities == 200:
                const.lr = 0.01
                const.epochs = 10
                const.pgu_gamma = 0.75
                const.use_train_aug = False
        elif tmp_args.dataset == "celeba":
            if tmp_args.num_identities <= 20:
                const.lr = 0.1
                const.epochs = 5
                const.pgu_gamma = 0.9
                const.use_train_aug = False
            elif tmp_args.num_identities == 50:
                const.lr = 0.1
                const.epochs = 5
                const.pgu_gamma = 0.9
                const.use_train_aug = False
            elif tmp_args.num_identities == 200:
                const.lr = 0.01
                const.epochs = 10
                const.pgu_gamma = 0.75
                const.use_train_aug = False
        elif tmp_args.dataset == "mufac":
            if tmp_args.num_identities == 5:
                const.lr = 0.01
                const.epochs = 5
                const.pgu_gamma = 0.9
                const.use_train_aug = False
            elif tmp_args.num_identities == 10:
                const.lr = 0.1
                const.epochs = 10
                const.pgu_gamma = 0.75
                const.use_train_aug = False

    if tmp_args.method == "loss_learning":
        const.batch_size = 64
        if tmp_args.dataset != "mufac":
            const.epochs = 3
            const.lr = 1e-4
            const.meta_lr = 0.1
        else:
            if tmp_args.num_identities == 5:
                const.epochs = 10
                const.lr = 1e-2
                const.meta_lr = 1e-3
            elif tmp_args.num_identities == 10:
                const.epochs = 3
                const.lr = 1e-3
                const.meta_lr = 1e-3
        const.optimizer = "adamw"
        const.scheduler = "cosine"
        const.warmup_epochs = 0
        const.num_identities_simulation = tmp_args.num_identities
        if tmp_args.num_identities == 1:
            const.num_identities_simulation = 50 if tmp_args.dataset == "celebahq" else 10
            const.lr = 1e-4 if tmp_args.dataset == "celebahq" else 1e-3

    if tmp_args.method == "scrub":
        const.epochs = 10
        const.optimizer = "sgd"
        const.momentum = 0.9
        const.weight_decay = 5e-4
        const.scheduler = "cosine"
        if tmp_args.dataset == "celeba":
            if tmp_args.num_identities in (1, 5, 10):
                const.lr = 5e-3
                const.use_train_aug = False
                const.forgetting_epochs = 5
            elif tmp_args.num_identities == 200:
                const.lr = 1e-3
                const.use_train_aug = False
                const.forgetting_epochs = 7
            elif tmp_args.num_identities == 500:
                const.lr = 1e-3
                const.use_train_aug = False
                const.forgetting_epochs = 3
        elif tmp_args.dataset == "celebahq":
            if tmp_args.num_identities in (1, 20, 50):
                const.lr = 1e-2
                const.forgetting_epochs = 5
            elif tmp_args.num_identities == 200:
                const.lr = 1e-3
                const.forgetting_epochs = 5
            elif tmp_args.num_identities == 500:
                const.lr = 5e-4
                const.forgetting_epochs = 7
        elif tmp_args.dataset == "mufac":
            if tmp_args.num_identities == 5:
                const.lr = 1e-3
                const.use_train_aug = False
                const.forgetting_epochs = 7
            elif tmp_args.num_identities == 10:
                const.lr = 1e-3
                const.use_train_aug = True
                const.forgetting_epochs = 5


    if tmp_args.method == "bad_teacher":
        const.epochs = 5
        const.optimizer = "sgd"
        const.momentum = 0.9
        const.weight_decay = 5e-4
        const.scheduler = "cosine"
        const.use_train_aug = False
        if tmp_args.dataset == "celeba":
            if tmp_args.num_identities in (1, 5, 10):
                const.lr = 1e-4
            elif tmp_args.num_identities == 200:
                const.batch_size = 256
                const.lr = 1e-4
            elif tmp_args.num_identities == 500:
                const.batch_size = 256
                const.lr = 1e-4
        elif tmp_args.dataset == "celebahq":
            if tmp_args.num_identities in (1, 20, 50):
                const.lr = 1e-4
            elif tmp_args.num_identities == 200:
                const.batch_size = 256
                const.lr = 1e-4
            elif tmp_args.num_identities == 500:
                const.batch_size = 256
                const.lr = 1e-5
        elif tmp_args.dataset == "mufac":
            if tmp_args.num_identities == 5:
                const.lr = 1e-3
                const.use_train_aug = True
            elif tmp_args.num_identities == 10:
                const.lr = 1e-3
                const.use_train_aug = True

    return const
