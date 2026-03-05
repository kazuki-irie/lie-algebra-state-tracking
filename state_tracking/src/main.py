"""Main entry point for training models."""

import json
import logging
import math
import hashlib
import warnings

import os
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from pprint import pformat
from random import randint
import shutil

import fire
import humanize
import polars as pl
import pyrootutils
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from ordered_set import OrderedSet
from sfirah.metrics import (
    ce_loss,
    detach_and_pad,
    reduce_metrics,
    sequence_accuracy,
    token_accuracy,
)
# from mamba import MambaWP
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from utils import cumulative_sequence_accuracies
import wandb

try:
    from fla.models import GatedDeltaProductConfig, GLAConfig, TransformerConfig
    from src.model import GatedDeltaProduct, MambaLSTM, GLA, VanillaTransformer
except ImportError:
    warnings.warn("FLA Models not available", RuntimeWarning)

try:
    from mamba_ssm.models.config_mamba import MambaConfig
except ImportError:
    warnings.warn("Mamba not available", RuntimeWarning)

try:
    from src.model import AUSSM, AussmConfig
except ImportError:
    AUSSM = None
    warnings.warn("AUSSM not available", RuntimeWarning)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

load_dotenv()
date_str = datetime.now().strftime("%Y-%m-%d")


def save_pretrained(accelerator, model, group, model_name, k, num_householders, seed, save_root="checkpoints"):
    """
    Save the unwrapped model weights and config to a dynamically generated directory.
    """
    # Generate a dynamic save directory name
    save_dir = os.path.join(
        save_root,
        f"{group}_{model_name}_k{k}_householders{num_householders}_{date_str}_{seed}"
    )

    os.makedirs(save_dir, exist_ok=True)

    # 1. Unwrap the model if using Accelerate
    unwrapped_model = accelerator.unwrap_model(model)

    # 2. Save the config as JSON
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(unwrapped_model.config.to_dict(), f, indent=2)

    # 3. Save the state_dict
    weights_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(unwrapped_model.state_dict(), weights_path)

    print(f"Model and config saved to {save_dir}")


class SpecialTokens(StrEnum):
    """Special tokens for tokenizer."""

    PAD = "[PAD]"
    BOS = "[BOS]"
    UNK = "[UNK]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    CLS = "[CLS]"
    MASK = "[MASK]"

    @classmethod
    def values(cls):
        """Return a list of the string values of each special token."""
        return list(map(lambda c: c.value, cls))

    @property
    def index(self):
        """Return the index of the token in the vocabulary.

        Used to get the index of the PAD token when directly modifying tensors.
        """
        return SpecialTokens.values().index(self.value)


def pad_collate(
        samples: list[dict[str, Tensor]], pad_token_id: int
) -> dict[str, Tensor]:
    """Collate function for DataLoader.

    Performs channel-wise padding of the inputs and targets.
    """
    # Only pad `labels` if len(labels) > 1,
    channels_to_pad = ["input_ids"]
    if samples[0]["labels"].dim() > 0:
        channels_to_pad.append("labels")

    max_lens = {}
    for c in channels_to_pad:
        max_lens[c] = max([s[c].shape[0] for s in samples])

    for s in samples:
        for c in channels_to_pad:
            if max_lens[c] > s[c].shape[0]:
                s[c] = F.pad(s[c], (0, max_lens[c] - s[c].shape[0]), value=pad_token_id)

    collated = {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
    }

    return collated


def tokenize(
        example: dict[str, Tensor],
        tokenizer: PreTrainedTokenizerFast,
        supervised: bool,
) -> dict[str, Tensor]:
    """Tokenize inputs."""
    tokenized = tokenizer(
        example["input"],
        return_tensors="pt",
        padding=True,
    )
    tokenized.pop("attention_mask", None)

    # If output is not supervised (e.g., for MLPs) then we only keep the final target
    # value since its sequence classification, not token classification.
    tokenized["labels"] = tokenizer(
        example["target"],
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    if not supervised:
        tokenized["labels"] = tokenized["labels"][:, -1]

    return tokenized


def get_dataset(
        group: str,
        k: int,
        strict_len: bool,
        train_size: float,
        data_dir: str | Path,
        supervised: bool = True,
        max_samples: int | None = None,
) -> dict:
    """Construct dataset."""
    assert train_size > 0 and train_size <= 1, "`train_size` must be in (0,1]"

    if strict_len:
        assert k > 1, "`k` must be at least 2"
        data_paths = [data_dir / f"{group}={i}.csv" for i in [k]]
        data_paths = list(OrderedSet(data_paths))
        # if not data_paths[0].exists():
        #     raise FileNotFoundError(f"You must have data for {group}={2}.")
        if not data_paths[-1].exists():
            raise FileNotFoundError(f"You must have data for {group}={k}.")
        log.info("Constructing dataset from:")
        log.info("  " + "\n  ".join(map(str, data_paths)))
    else:
        data_paths = [data_dir / f"{group}={i}.csv" for i in range(2, k + 1)]
        if not data_paths[0].exists():
            raise FileNotFoundError(f"You must have data for {group}=2.")
        if not data_paths[-1].exists():
            raise FileNotFoundError(f"You must have data for {group}={k}.")
        data_paths = [p for p in data_paths if p.exists()]
        data_paths = list(OrderedSet(data_paths))
        log.info("Constructing dataset from:")
        log.info("  " + "\n  ".join(map(str, data_paths)))

    # All unique tokens can be found by looking at the k=2 inputs. We create a
    # a dictionary mapping each token to its index in the vocabulary and use this
    # to construct the tokenizer.
    if group.startswith("S5_only_swaps"):
        # use the full S5=2 file to have the unique tokens
        unique_tokens = (
            pl.read_csv(data_dir / "S5=2.csv")
            .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
            .explode("input")
            .unique()["input"]
            .to_list()
        )
    elif "limit_to" in group:
        g = group[:2]
        # use the full S5=2 file to have the unique tokens
        unique_tokens = (
            pl.read_csv(data_dir / f"{g}=2.csv")
            .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
            .explode("input")
            .unique()["input"]
            .to_list()
        )
    else:
        unique_tokens = (
            pl.read_csv(data_paths[0])
            .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
            .explode("input")
            .unique()["input"]
            .to_list()
        )
    unique_tokens = {t: int(t) for t in unique_tokens}

    tokenizer_base = Tokenizer(WordLevel())
    tokenizer_base.pre_tokenizer = WhitespaceSplit()
    tokenizer_base.add_tokens(sorted(list(unique_tokens.keys()), key=lambda x: int(x)))
    tokenizer_base.add_special_tokens(SpecialTokens.values())
    tokenizer_base.post_processor = TemplateProcessing(
        single=f"{SpecialTokens.BOS} $A",
        special_tokens=[
            (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS))
        ],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_base,
        bos_token=SpecialTokens.BOS.value,
        unk_token=SpecialTokens.UNK.value,
        eos_token=SpecialTokens.EOS.value,
        sep_token=SpecialTokens.SEP.value,
        cls_token=SpecialTokens.CLS.value,
        mask_token=SpecialTokens.MASK.value,
        pad_token=SpecialTokens.PAD.value,
    )
    tokenizer.padding_side = "right"
    tokenize_map = partial(tokenize, tokenizer=tokenizer, supervised=supervised)

    # Construct dataset
    if len(data_paths) == 1:
        dataset = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )
        if max_samples is not None:
            num_samples = min(len(dataset), max_samples)
            dataset = dataset.select(range(num_samples))
        if train_size < 1:
            dataset = dataset.train_test_split(train_size=train_size)
    else:
        train_data = [
            load_dataset("csv", data_files=str(d_path), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
            for d_path in data_paths[:-1]
        ]
        k_data = (
            load_dataset("csv", data_files=str(data_paths[-1]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )

        if max_samples is not None:
            k_data = k_data.select(range(min(len(k_data), max_samples)))
            train_data = [t.select(range(min(len(t), max_samples))) for t in train_data]

        train_data = concatenate_datasets(train_data)

        if train_size < 1:
            dataset = k_data.train_test_split(train_size=train_size)
            dataset["train"] = concatenate_datasets([dataset["train"], train_data])
        else:
            dataset = concatenate_datasets([train_data, k_data])

    return {
        "dataset": dataset.with_format("torch"),
        "tokenizer": tokenizer,
        "n_vocab": tokenizer_base.get_vocab_size(with_added_tokens=True),
    }


def compute_metrics(
        data: list[(Tensor, Tensor)],
        tokenizer: PreTrainedTokenizerFast,
        metric_fns: dict[str, Callable] = {
            "loss": ce_loss,
            "token_accuracy": token_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "sequence_accuracies": cumulative_sequence_accuracies,
        },
        prefix: str | None = None,
) -> dict:
    """Compute metrics."""
    values_dict = {}

    data = detach_and_pad(data, pad_token_id=tokenizer.pad_token_id)
    predicted_logits = data["predictions"]
    target_tokens = data["targets"]

    prefix_str = "" if prefix is None else f"{prefix}/"
    for metric_name, metric_fn in metric_fns.items():
        values_dict[prefix_str + metric_name] = metric_fn(
            predicted_logits, target_tokens, tokenizer.pad_token_id
        )

    return values_dict


def make_job_name(
    hps: dict,
    priority=("model_name", "group", "n_layers"),
    exclude=(),
):
    parts = []

    # priority keys (in given order)
    for k in priority:
        if k in hps and k not in exclude:
            parts.append(f"{hps[k]}")

    # remaining keys (sorted)
    for k in sorted(hps):
        if k in priority or k in exclude:
            continue
        parts.append(f"{k}{hps[k]}")

    return "-".join(parts)


def check_nan(name, t):
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(
            f"{name} has NaN/Inf: "
            f"nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()} "
            f"min={t.min().item()} max={t.max().item()}")


def max_prefix_at_threshold(seq_accs, threshold):
    seq_list = list(seq_accs) if seq_accs is not None else []
    max_len = 0
    for i, acc in enumerate(seq_list):
        if float(acc) >= threshold:
            max_len = i + 1  # 1-indexed prefix length
    return max_len


def train(
        # Data parameters
        group: str,
        k: int,
        k_test: int = None,
        curriculum: bool = False,
        data_dir: Path = PROJECT_ROOT / "data",
        strict_len: bool = True,
        train_size: float = 0.99,
        tagging: bool = True,
        model_name: str = "deltaproduct",
        max_samples: int | None = None,
        # Model parameters
        hidden_size: int = 128,
        d_state: int = 16,  # for SSMs
        mamba_expand_rate: int = 2,
        # Simple one layer recursion params
        low_rank: int = 1,
        delta_rule_keys_activation: str = 'id',  # activation for delta rule, original is silu
        activation_diag: str = 'id',  # activation for the diagonal matrix case
        delta_rule_step_size_activation: str = 'sigmoid',
        # activation for the step size of  delta rule, original is sigmoid
        decoder_mode: str = "v1",
        normalize_recursion_output: str = True,
        bias: bool = True,
        # FLA model parameters
        n_layers: int = 1,
        allow_neg_eigval: bool = False,
        n_heads: int = 4,
        head_dim: int = 32,
        num_householder: int = 1,
        learn_init_state: bool = False,
        # Training parameters
        epochs: int = 500,
        batch_size: int = 2048,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        op_eps: float = 1e-8,
        weight_decay: float = 0.000001,
        compile: bool = False,
        gradient_clip: float | None = None,
        max_val_acc: float | None = 0.99,
        use_scheduler: bool = False,
        # Misc
        log_level: str = "INFO",
        seed: int = randint(0, 2 ** 32 - 1),
        project_name: str = "word_problem_linrnnsimple",
        logging: bool = True,
        wandb_user: str = 'irie',
        checkpoint_root: str = "run_checkpoints",
        save_every_steps: int = 1000,
        resume: bool = True,
        enable_mix_precision: bool = False,
):
    """Train Simple RNN model."""
    set_seed(seed)

    accelerator = Accelerator(log_with="wandb") if logging else Accelerator()
    log.setLevel(log_level)

    # Load dataset
    datadict = get_dataset(
        group=group,
        k=k,
        strict_len=strict_len,
        train_size=train_size,
        data_dir=data_dir,
        supervised=tagging,
        max_samples=max_samples,
    )

    dataset = datadict["dataset"]
    n_vocab = datadict["n_vocab"]
    tokenizer = datadict["tokenizer"]
    collate_fn = partial(pad_collate, pad_token_id=tokenizer.pad_token_id)

    # Load test dataset (possibly different length)
    if k_test is None or k_test == k:
        dataset_test = dataset
    else:
        datadict = get_dataset(
            group=group,
            k=k_test,
            strict_len=strict_len,
            train_size=train_size,
            data_dir=data_dir,
            supervised=tagging,
            max_samples=max_samples,
        )
        dataset_test = datadict["dataset"]

    # Set up logger
    project_hps = {
        "batch_size": batch_size,
        "betas": (beta1, beta2),
        "bias": bias,
        "compile": compile,
        "model_name": model_name,
        "hidden_size": hidden_size,
        "d_state": d_state,
        "mamba_expand_rate": mamba_expand_rate,
        "low_rank": low_rank,
        "n_train_samples": dataset['train'].num_rows,
        "n_test_samples": dataset_test['test'].num_rows,
        "activation_diag": activation_diag,
        "delta_rule_keys_activation": delta_rule_keys_activation,
        "delta_rule_step_size_activation": delta_rule_step_size_activation,
        "n_layers": n_layers,
        "allow_neg_eigval": allow_neg_eigval,
        "headim": head_dim,
        "num_householder": num_householder,
        "n_heads": n_heads,
        "decoder_mode": decoder_mode,
        "normalize_recursion_output": normalize_recursion_output,
        "enable_mix_precision": enable_mix_precision,
        "use_scheduler": use_scheduler,
        "epochs": epochs,
        "eps": op_eps,
        "group": group,
        "gradient_clip": gradient_clip,
        "k": k,
        "k_test": k_test,
        "lr": lr,
        "max_val_acc": max_val_acc,
        "max_samples": max_samples,
        "n_vocab": n_vocab,
        "seed": seed,
        "strict_len": strict_len,
        "tagging": tagging,
        "train_size": train_size,
        "weight_decay": weight_decay,
        "curriculum": curriculum,
        "learn_init_state": learn_init_state,
    }

    # create run name for wandb
    exp_str = ''
    for arg_key in project_hps.keys():
        exp_str += str(project_hps[arg_key]) + '-'

    # taken from https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    exp_hash = str(int(hashlib.sha1(exp_str.encode("utf-8")).hexdigest(), 16) % (10 ** 8))

    job_name = make_job_name(
        project_hps,
        priority=("model_name", "group", "n_layers", "hidden_size", "lr", "seed", "k", "k_test", "mamba_expand_rate"),
        exclude={
            "tagging", "betas", "bias", "compile", "low_rank",
            "activation_diag", "curriculum", "strict_len", "eps",
            "normalize_recursion_output", "enable_mix_precision",
            "decoder_mode", "delta_rule_keys_activation",
            "delta_rule_step_size_activation",
            "use_scheduler", "max_samples",
            "train_size", "allow_neg_eigval",
            "n_test_samples", "n_train_samples",
            "max_val_acc"},
    )

    wandb_run_name = f"{job_name}-{exp_hash}"

    run_ckpt_dir = os.path.join(checkpoint_root, wandb_run_name, "latest")
    state_path = os.path.join(run_ckpt_dir, "trainer_state.json")
    has_ckpt = resume and os.path.isdir(run_ckpt_dir) and os.path.exists(state_path)

    # os.makedirs(run_ckpt_dir, exist_ok=True)

    accelerator.init_trackers(
        project_name,
        config=project_hps,
        init_kwargs={"wandb": {
            "entity": f'{wandb_user}',
            "name": wandb_run_name},}
    )
    log.info(f"Wandb run name: {wandb_run_name}")

    log.info(f"Config: {pformat(project_hps)}")
    log.info(f"Dataset (k={k}): {dataset}")
    log.info(f"Dataset test (k={k_test}): {dataset_test}")

    # Construct model
    use_aussm = False
    if tagging:
        if model_name == "deltaproduct":
            print(f'Launching DeltaProduct-{num_householder} with {n_layers} layers and {n_heads} heads')
            conf = GatedDeltaProductConfig(
                vocab_size=n_vocab,
                use_short_conv=False,
                use_gate=False,  # no gating
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                use_forget_gate=False,
                num_heads=n_heads,
                expand_v=1,
                bos_token_id=0, eos_token_id=n_vocab - 1,
                fuse_cross_entropy=False,
                allow_neg_eigval=allow_neg_eigval,
                head_dim=head_dim,
                learn_init_state=learn_init_state,
                num_householder=num_householder,
            )
            model = GatedDeltaProduct(conf)
        # kazuki: for any models below, I hope I didn't overlook any important hyper-params!
        elif model_name == "gla":
            print(f'Launching GLA with {n_layers} layers and {n_heads} heads')
            conf = GLAConfig(
                vocab_size=n_vocab,
                use_short_conv=False,
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_heads=n_heads,
                expand_k=1,
                expand_v=1,
                bos_token_id=0, eos_token_id=n_vocab - 1,
                learn_init_state=learn_init_state,
                fuse_cross_entropy=False,
            )
            model = GLA(conf)
        elif model_name == "transformer":
            print(f'Launching Vanilla Transformer with {n_layers} layers and {n_heads} heads')
            conf = TransformerConfig(
                vocab_size=n_vocab,
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_heads=n_heads,
                bos_token_id=0, eos_token_id=n_vocab - 1,
                fuse_cross_entropy=False,
            )
            model = VanillaTransformer(conf)
        elif model_name == "mamba_lstm":
            print(f'Launching MambaLSTM with {n_layers} layers and {n_heads} heads')
            conf = MambaConfig(
                vocab_size=n_vocab,
                d_model=hidden_size,
                n_layer=n_layers,
                ssm_cfg={
                    "d_state": d_state,
                    "expand": mamba_expand_rate,
                    # convolution kernel size: common default is 4.
                    "d_conv": 4,
                    "positive_and_negative_associative_scan": False,
                },
            )  # I'm leaving conv for Mamba for now as there is no option to remove it without touching the code.
            model = MambaLSTM(conf)
        elif model_name == "neg_mamba_lstm":
            print(f'Launching PosNegative MambaLSTM with {n_layers} layers')
            conf = MambaConfig(
                vocab_size=n_vocab,
                d_model=hidden_size,
                n_layer=n_layers,
                ssm_cfg={
                    "d_state": d_state,
                    "expand": mamba_expand_rate,
                    # convolution kernel size: common default is 4.
                    "d_conv": 4,
                    "positive_and_negative_associative_scan": True,
                },
            )  # I'm leaving conv for Mamba for now as there is no option to remove it without touching the code.
            model = MambaLSTM(conf)
        elif model_name == "aussm":
            use_aussm = True
            print(f'Launching AUSSM with {n_layers} layers')
            assert n_layers > 0
            # AUSSM assumes str-encoded layer specification a for AUSSM (m for mamba)
            layer_str = 'a'
            for _ in range(n_layers-1):
                layer_str += '|a'
            conf = AussmConfig(
                d_model=hidden_size,
                vocab_size=n_vocab,
                output_vocab_size=n_vocab,
                layers=layer_str,
                d_state=d_state,
                mamba_expand=mamba_expand_rate,
                dt_rank='auto',
                d_conv=4,
                conv_bias=True,
                bias=False,
                ssmau_conv_1d=False,
                embedding_decay=False,
                ssmau_cuda=True,
                verbose=True)
            model = AUSSM(conf)
        else:
            raise ValueError(f"model_name: {model_name} not supported.")
        # if enable_mix_precision:
        #     assert not use_aussm, "incompatible with AUSSM using complex numbers"
        if not use_aussm:
            model.to(torch.bfloat16)
        model.to('cuda')

    if compile:
        torch.set_float32_matmul_precision('high')
        log.info("Compiling model...")
        model = torch.compile(model, dynamic=True)
        log.info("Model compiled!")

    log.info(f"Model: {model}")
    log.info(
        f"Number of parameters: {humanize.intword(model.num_parameters)}"
        f" ({model.num_parameters})"
    )
    log.info(f"Accelerator state: {accelerator.state}")

    device = accelerator.device

    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=op_eps,
        weight_decay=weight_decay,
    )

    def inverse_sqrt_scheduler(epoch):
        return 1 / math.sqrt(epoch + 1)  # Add 1 to avoid division by zero at epoch 0

    if train_size < 1:
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        eval_dataloader = DataLoader(
            dataset_test["test"],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        eval_dataloader = DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    # Set up the scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_dataloader))
    model, optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, eval_dataloader
    )

    metric_fns = {
        "loss": ce_loss,
        "sequence_accuracy": token_accuracy,
    }

    if tagging:
        metric_fns["sequence_accuracy"] = sequence_accuracy
        metric_fns["token_accuracy"] = token_accuracy
        metric_fns["sequence_accuracies"] = cumulative_sequence_accuracies

    global_step = 0
    best_val_acc = 0.0
    start_epoch = 0

    THRESHOLD_90 = 0.90
    THRESHOLD_99 = 0.99

    best_train_maxlen_at_90 = 0
    best_val_maxlen_at_90 = 0

    best_train_maxlen_at_99 = 0
    best_val_maxlen_at_99 = 0

    # Keep a reference to the original dataloader
    train_dataloader_base = train_dataloader

    curriculum_idx = (min(2, k + 1) if curriculum else k + 1)
    resume_step_in_epoch = 0
    if has_ckpt:
        accelerator.print(f"Resuming from checkpoint: {run_ckpt_dir}")
        accelerator.load_state(run_ckpt_dir)

        with open(state_path, "r") as f:
            st = json.load(f)

        global_step = int(st.get("global_step", 0))
        best_val_acc = float(st.get("best_val_acc", 0.0))
        curriculum_idx = int(st.get("curriculum_idx", curriculum_idx))
        start_epoch = int(st.get("epoch", global_step // len(train_dataloader)))

        steps_per_epoch = len(train_dataloader)
        resume_step_in_epoch = global_step % steps_per_epoch

    print(f"Starting training  (start_curriculum_length={curriculum_idx}, k={k})")
    try:
        for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", position=0, leave=False):
            model.train()

            # Only skip batches in the first resumed epoch
            if epoch == start_epoch and resume_step_in_epoch > 0:
                epoch_dataloader = accelerator.skip_first_batches(
                    train_dataloader_base, resume_step_in_epoch
                )
                accelerator.print(f"Epoch {epoch}: skipping first {resume_step_in_epoch} batches")
            else:
                epoch_dataloader = train_dataloader_base

            train_results = []
            epoch_train_loss = 0
            n_batches = 0
            for batch in (
                    t_bar := tqdm(epoch_dataloader, desc="Train", position=1, leave=False)
            ):
                global_step += 1
                optimizer.zero_grad()
                source = batch["input_ids"]
                target = batch["labels"]
                if use_aussm:
                    slen = source.shape[-1]
                    if slen % 32 != 0:
                        # hard coded for AUSSM; backward kernel requires nice round len
                        assert (slen - 1) % 32 == 0, f'slen = {slen}'
                        source = source[:, :slen-1]
                        target = target[:, :slen-1]
                        warnings.warn(f"[AUSSM] Seq len is {slen}, cut it to {slen - 1}", RuntimeWarning)


                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=enable_mix_precision):
                    output = model(source)
                    out_used = output[:, :curriculum_idx, :]
                    tgt_used = target[:, :curriculum_idx]

                    # gather for metrics
                    preds_used, refs_used = accelerator.gather_for_metrics((out_used, tgt_used))

                    # compute metrics dict in one place
                    metrics = compute_metrics(
                        [(preds_used, refs_used)],
                        tokenizer=tokenizer,
                        metric_fns=metric_fns,
                        prefix="train",
                    )

                    train_results.append(metrics)

                    loss = F.cross_entropy(
                        out_used.flatten(end_dim=-2),
                        tgt_used.flatten(),
                    )

                    check_nan("loss", loss)

                epoch_train_loss += loss.item()
                n_batches += 1

                accelerator.backward(loss)
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip, norm_type=2.0
                    )
                optimizer.step()

                if use_scheduler:
                    scheduler.step()

                should_save = (global_step % save_every_steps == 0)
                # ---------------- NEW: save latest checkpoint (overwrite) ----------------
                if accelerator.is_main_process and should_save:
                    os.makedirs(os.path.dirname(run_ckpt_dir), exist_ok=True)

                    tmp_dir = run_ckpt_dir + ".tmp"
                    bak_dir = run_ckpt_dir + ".bak"

                    # clean tmp if leftover
                    if os.path.isdir(tmp_dir):
                        shutil.rmtree(tmp_dir)

                    accelerator.save_state(tmp_dir)

                    # write trainer_state.json into tmp_dir first
                    tmp_state_path = os.path.join(tmp_dir, "trainer_state.json")
                    with open(tmp_state_path, "w") as f:
                        json.dump(
                            {
                                "global_step": global_step,
                                "epoch": epoch,
                                "best_val_acc": best_val_acc,
                                "curriculum_idx": curriculum_idx,
                            },
                            f,
                            indent=2,
                        )

                    # safe swap: keep old until new is in place
                    try:
                        if os.path.isdir(bak_dir):
                            shutil.rmtree(bak_dir)
                        if os.path.isdir(run_ckpt_dir):
                            os.rename(run_ckpt_dir, bak_dir)
                        os.rename(tmp_dir, run_ckpt_dir)
                        if os.path.isdir(bak_dir):
                            shutil.rmtree(bak_dir)
                    except Exception:
                        # restore previous latest if swap failed
                        if os.path.isdir(run_ckpt_dir):
                            shutil.rmtree(run_ckpt_dir, ignore_errors=True)
                        if os.path.isdir(bak_dir):
                            os.rename(bak_dir, run_ckpt_dir)
                        raise
                    finally:
                        if os.path.isdir(tmp_dir):
                            shutil.rmtree(tmp_dir, ignore_errors=True)

                if should_save:
                    accelerator.wait_for_everyone()

            epoch_train_loss = epoch_train_loss / max(n_batches, 1)

            accelerator.log({"epoch": epoch, "curriculum_idx": curriculum_idx}, step=global_step)
            accelerator.log(reduce_metrics(train_results), step=global_step)

            if epoch_train_loss < 0.3 and curriculum_idx < k + 1:
                curriculum_idx = curriculum_idx + 1
                print(f' Increasing curriculum index to {curriculum_idx}')

            model.eval()
            eval_results = []
            for batch in tqdm(eval_dataloader, desc="Eval", position=1, leave=False):
                source = batch["input_ids"]
                target = batch["labels"]
                if use_aussm:
                    slen = source.shape[-1]
                    if slen % 32 != 0:
                        assert (slen - 1) % 32 == 0, f'slen = {slen}'
                        source = source[:, :slen-1]
                        target = target[:, :slen-1]
                        warnings.warn(f"[AUSSM] Seq len is {slen}, cut it to {slen - 1} ", RuntimeWarning)

                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=enable_mix_precision):
                        output = model(source)

                predictions, references = accelerator.gather_for_metrics((output, target))

                eval_results.append(
                    compute_metrics(
                        [(predictions, references)],
                        prefix="val",
                        tokenizer=tokenizer,
                        metric_fns=metric_fns,
                    )
                )

            eval_metrics = reduce_metrics(eval_results)
            train_metrics = reduce_metrics(train_results)

            # ---------------- scalar summary: max sequence length with >= THRESHOLD accuracy ----------------
            # current epoch values
            train_maxlen_at_90 = max_prefix_at_threshold(
                train_metrics.get("train/sequence_accuracies", []),
                THRESHOLD_90,
            )
            val_maxlen_at_90 = max_prefix_at_threshold(
                eval_metrics.get("val/sequence_accuracies", []),
                THRESHOLD_90,
            )

            train_maxlen_at_99 = max_prefix_at_threshold(
                train_metrics.get("train/sequence_accuracies", []),
                THRESHOLD_99,
            )
            val_maxlen_at_99 = max_prefix_at_threshold(
                eval_metrics.get("val/sequence_accuracies", []),
                THRESHOLD_99,
            )

            # log current (per-epoch) scalars
            accelerator.log(
                {
                    "train/max_seq_len_at_90": int(train_maxlen_at_90),
                    "val/max_seq_len_at_90": int(val_maxlen_at_90),
                    "train/max_seq_len_at_99": int(train_maxlen_at_99),
                    "val/max_seq_len_at_99": int(val_maxlen_at_99),
                },
                step=global_step,
            )

            # update best-so-far (over course of training)
            if train_maxlen_at_90 > best_train_maxlen_at_90:
                best_train_maxlen_at_90 = int(train_maxlen_at_90)

            if val_maxlen_at_90 > best_val_maxlen_at_90:
                best_val_maxlen_at_90 = int(val_maxlen_at_90)

            if train_maxlen_at_99 > best_train_maxlen_at_99:
                best_train_maxlen_at_99 = int(train_maxlen_at_99)

            if val_maxlen_at_99 > best_val_maxlen_at_99:
                best_val_maxlen_at_99 = int(val_maxlen_at_99)

            # log best-so-far scalars (these are what you want to watch in W&B)
            accelerator.log(
                {
                    "best/train_max_seq_len_at_90": int(best_train_maxlen_at_90),
                    "best/val_max_seq_len_at_90": int(best_val_maxlen_at_90),
                    "best/train_max_seq_len_at_99": int(best_train_maxlen_at_99),
                    "best/val_max_seq_len_at_99": int(best_val_maxlen_at_99),
                },
                step=global_step,
            )
            # ---------------- TRAIN sequence accuracy table ----------------
            if "train/sequence_accuracies" in train_metrics:
                train_seq_vals = train_metrics["train/sequence_accuracies"]

                train_data = [
                    [i + 1, acc] for i, acc in enumerate(train_seq_vals)
                ]

                train_table = wandb.Table(
                    data=train_data,
                    columns=["sequence_length", "accuracy"],
                )

                accelerator.log(
                    {
                        "train_sequence_accuracy_table": wandb.plot.line(
                            train_table,
                            "sequence_length",
                            "accuracy",
                            title="Train sequence accuracy",
                        )
                    },
                    step=global_step,
                )

            stats = [
                dict(
                    name="val_sequence_accuracy",
                    values=eval_metrics["val/sequence_accuracies"],
                    x_label="sequence_length",
                    y_label="accuracy",
                ),
            ]

            model_statistics = model.get_useful_stats()

            stats.extend(
                [dict(name=s_name, values=s, x_label="tokens", y_label=s_name) for s_name, s in
                 model_statistics.items()])

            for s in stats:
                data = [[x, y] for (x, y) in zip(range(0, len(s['values'])), s['values'])]
                table = wandb.Table(data=data, columns=[s["x_label"], s["y_label"]])
                s["table"] = table

            for s in stats:
                accelerator.log({f"{s['name']}": wandb.plot.line(
                    s["table"], s["x_label"], s["y_label"], title=""
                )}, step=global_step)
            if eval_metrics["val/sequence_accuracy"] > best_val_acc:
                for s in stats:
                    accelerator.log({f"best_{s['name']}": wandb.plot.line(
                        s["table"], s["x_label"], s["y_label"], title=""
                    )}, step=global_step)

                best_val_acc = eval_metrics["val/sequence_accuracy"]
                save_pretrained(
                    accelerator=accelerator,
                    model=model,
                    group=group,
                    model_name=model_name,  # Dynamically use the class name
                    k=k,
                    num_householders=num_householder,  # Or whatever variable you're using
                    save_root="checkpoints",
                    seed=seed
                )
            eval_metrics["val/best_sequence_accuracy"] = best_val_acc
            accelerator.log(eval_metrics, step=global_step)

            if max_val_acc is not None and best_val_acc >= max_val_acc:
                log.info(f"Validation accuracy reached {max_val_acc}. Stopping training.")
                break

        log.info(eval_metrics)
        accelerator.end_training()
    except Exception as e:
        # Log the exception details to wandb
        wandb.alert(
            title="Training Error",
            text=f"An error occurred: {str(e)}"
        )
        # Optionally log the traceback
        import traceback
        wandb.log({"error_traceback": traceback.format_exc()})
        # Reraise the exception if needed
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire()
