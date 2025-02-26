import argparse
from method.configs.const import get_const


def str2bool(v):
    """Converts a string to a boolean value."""
    return v.lower() in ("true", "t", "y", "yes", "1")


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # --- Logging info ---#
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--description", type=str, default="")

    # --- Model Hyperparameters ---#
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        choices=[
            "vit_base_patch16_224",
            "resnet18", # we never used it, but should work
        ],
    )
    parser.add_argument("--pretrained", type=str2bool, default="True")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="checkpoints/", 
        help="Directory where model chekpoints are stored"
    )

    parser.add_argument("--hidden_size", type=int, default=512, help="MetaLoss hidden size")
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=0, 
        help="MetaLoss depth. In our experiments we set it to 0."
    )
    parser.add_argument("--identity_embed_size", type=int, default=64)
    parser.add_argument("--prob_dropout", type=float, default=0.5)
    parser.add_argument(
        "--use_retain", 
        type=str2bool, 
        default="False", 
        help="Additionally aligns model with retain set. See paper Appendix for further details. Default to False."
    )
    parser.add_argument(
        "--forget_epochs", 
        type=int, 
        default=1, 
        help="Number of unlearning steps. One is enough to achieve the best results"
    )
    parser.add_argument("--forget_loss", default="smooth_l1", choices=["smooth_l1", "l1", "l2"])
    parser.add_argument(
        "--loss_type",
        default="full",
        choices=["full", "original", "unlearned", "scrub", "rev"],
        help="Type of loss function to use. original: computes alignment between forget and original validation. unlearned: computes alignment between forget and unlearned validation. full: combines both. scrub: uses SCRUB loss. rev: is a version asked by a reviewer during the rebuttal phase (see OpenReview)",
    )
    parser.add_argument(
        "--use_accs", 
        type=str2bool, 
        default="True", 
        help="Scales loss by opposite of accuracy. See main paper."
    )
    parser.add_argument(
        "--use_feats", 
        type=str2bool, 
        default="True", 
        help="Concatenates features as input to the meta-loss"
    )
    parser.add_argument(
        "--use_ids", 
        type=str2bool, 
        default="True", 
        help="Concatenates identity info as input to the meta-loss"
    )
    parser.add_argument(
        "--use_targets", 
        type=str2bool, 
        default="True", 
        help="Concatenates target vector as input to the meta-loss"
    )
    parser.add_argument(
        "--robustness", 
        action="store_true", 
        help="To reproduce Tab. 7 results."
    )

    # --- Dataset ---#
    parser.add_argument(
        "--dataset", type=str, choices=["celeba", "celebahq", "mufac"], default="celebahq"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="datasets/", 
        help="Directory where datasets are stored"
    )

    # --- Task Hyperparameters ---#
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "pretrain",
            "retrain",
            "scrub",
            "bad_teacher",
            "test_loss",
            "meta_unlearn",
            "ssd",
            "lipschitz",
            "pgu",
        ],
        default="pretrain",
    )
    parser.add_argument(
        "--num_identities", type=int, default=20, help="Number of identities to forget"
    )

    # --- Get Constants ---#
    tmp_args, _ = parser.parse_known_args()
    const = get_const(tmp_args)

    # --- Loss Learning Hyperparameters ---#
    parser.add_argument("--meta_lr", type=float, default=const.meta_lr)
    parser.add_argument(
        "--num_identities_simulation", type=int, default=const.num_identities_simulation
    )

    # --- SCRUB Hyperparameters ---#
    parser.add_argument("--alpha_scrub", type=float, default=0.001)
    parser.add_argument("--gamma_scrub", type=float, default=0.99)
    parser.add_argument("--forgetting_epochs", type=int, default=const.forgetting_epochs)
    parser.add_argument("--temperature_scrub", type=float, default=4)
    parser.add_argument("--batch_size_retain", type=int, default=32)
    parser.add_argument("--use_train_aug", type=str2bool, default=const.use_train_aug)

    # --- SSD Hyperparameters ---#
    parser.add_argument(
        "--ssd_dampening_constant", type=float, default=const.ssd_dampening_constant
    )
    parser.add_argument(
        "--ssd_selection_weighting", type=float, default=const.ssd_selection_weighting
    )

    # --- Lipschitz Hyperparameters ---#
    parser.add_argument("--lipschitz_weighting", type=float, default=const.lipschitz_weighting)
    parser.add_argument("--lipschitz_n_samples", type=int, default=25)

    # --- PGU Hyperparameters ---#
    parser.add_argument("--pgu_gamma", type=float, default=const.pgu_gamma)

    # --- Training Hyperparameters ---#
    parser.add_argument("--batch_size", type=int, default=const.batch_size)
    parser.add_argument("--epochs", type=int, default=const.epochs)
    parser.add_argument("--lr", type=float, default=const.lr)
    parser.add_argument("--clip_grad", type=float, default=const.clip_grad)
    parser.add_argument("--momentum", type=float, default=const.momentum)
    parser.add_argument("--weight_decay", type=float, default=const.weight_decay)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam", "adamw"],
        default=const.optimizer,
    )
    parser.add_argument(
        "--amsgrad",
        type=str2bool,
        default=const.amsgrad,
        help="Whether to use the AMSGrad variant of Adam",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["const", "step", "multistep", "cosine", "exp", "linear"],
        default=const.scheduler,
    )
    parser.add_argument(
        "--t_max",
        type=int,
        default=const.t_max,
        help="Number of epochs for cosine annealing scheduler",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=const.step_size,
        help="Number of epochs for step scheduler",
    )
    parser.add_argument(
        "--milestones",
        type=list,
        default=const.milestones,
        help="Milestones for multistep scheduler",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=const.gamma,
        help="Gamma for step and multistep scheduler",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=const.warmup_epochs,
        help="Number of epochs for warmup scheduler",
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=const.evaluate_every,
        help="Number of epochs to evaluate the model",
    )

    # --- DataLoader Hyperparameters ---#
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--drop_last", type=str2bool, default="False")
    parser.add_argument("--pin_memory", type=str2bool, default="True")

    # --- Misc ---#
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disables saving of checkpoints, logging, and set epochs to 1",
    )
    parser.add_argument("--save", type=str2bool, default="True")
    parser.add_argument("--store_dir", type=str, default="output/")

    # --- Logging ---#
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="unlearning")
    parser.add_argument("--entity", type=str, default="tdemin")
    parser.add_argument("--offline", action="store_true")

    return parser


def custom_bool(v):
    """Converts a string to a boolean value."""
    return v.lower() in ("true", "t", "y", "yes", "1")


def parse_args() -> argparse.Namespace:
    parser = get_argparser()
    args = parser.parse_args()

    if args.t_max > args.epochs:
        args.t_max = args.epochs

    if args.debug:
        args.epochs = 1
        args.name = "debug"
        args.project = "debug"

    return args
