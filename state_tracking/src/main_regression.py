import json
import logging
import math
import hashlib
import warnings
import os
import shutil

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from pprint import pformat
from random import randint

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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch import Tensor, optim, nn
from torch.utils.data import DataLoader, Dataset
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

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
load_dotenv()
date_str = datetime.now().strftime("%Y-%m-%d")


# =========================================================
# Saving
# =========================================================

def save_pretrained(accelerator, model, group, model_name, k, num_householders, seed, save_root="checkpoints"):
    """Save unwrapped model weights and config to a dynamically generated directory."""
    save_dir = os.path.join(
        save_root,
        f"{group}_{model_name}_k{k}_householders{num_householders}_{date_str}_{seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(unwrapped_model.config.to_dict(), f, indent=2)

    weights_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(unwrapped_model.state_dict(), weights_path)

    print(f"Model and config saved to {save_dir}")


# =========================================================
# Tokenizer
# =========================================================

class SpecialTokens(StrEnum):
    PAD = "[PAD]"
    BOS = "[BOS]"
    UNK = "[UNK]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    CLS = "[CLS]"
    MASK = "[MASK]"

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

    @property
    def index(self):
        return SpecialTokens.values().index(self.value)


# =========================================================
# Collators
# =========================================================

def pad_collate(samples: list[dict[str, Tensor]], pad_token_id: int) -> dict[str, Tensor]:
    """Collate for classification."""
    channels_to_pad = ["input_ids"]
    if samples[0]["labels"].dim() > 0:
        channels_to_pad.append("labels")

    max_lens = {c: max(int(s[c].shape[0]) for s in samples) for c in channels_to_pad}

    for s in samples:
        for c in channels_to_pad:
            if max_lens[c] > int(s[c].shape[0]):
                s[c] = F.pad(s[c], (0, max_lens[c] - int(s[c].shape[0])), value=pad_token_id)

    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
    }


def pad_collate_v0(samples: list[dict[str, Tensor]], pad_token_id: int) -> dict[str, Tensor]:
    """
    Collate for regression:
      input_ids: [B,T]
      mask:      [B,T] (True where valid)
      v0:        [B,3]
    """
    max_len = max(int(s["input_ids"].shape[0]) for s in samples)
    B = len(samples)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    v0 = torch.stack([s["v0"] for s in samples], dim=0)  # [B,3]

    for i, s in enumerate(samples):
        T = int(s["input_ids"].shape[0])
        input_ids[i, :T] = s["input_ids"].to(torch.long)
        mask[i, :T] = True

    return {"input_ids": input_ids, "mask": mask, "v0": v0}


# =========================================================
# Tokenization / Dataset loading
# =========================================================

def tokenize(example: dict[str, Tensor], tokenizer: PreTrainedTokenizerFast, supervised: bool) -> dict[str, Tensor]:
    tokenized = tokenizer(example["input"], return_tensors="pt", padding=True)
    tokenized.pop("attention_mask", None)

    tokenized["labels"] = tokenizer(example["target"], return_tensors="pt", padding=True)["input_ids"]
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
    assert 0 < train_size <= 1, "`train_size` must be in (0,1]"

    if strict_len:
        assert k > 1, "`k` must be at least 2"
        data_paths = [Path(data_dir) / f"{group}={k}.csv"]
        data_paths = list(OrderedSet(data_paths))
        if not data_paths[-1].exists():
            raise FileNotFoundError(f"You must have data for {group}={k}.")
    else:
        data_paths = [Path(data_dir) / f"{group}={i}.csv" for i in range(2, k + 1)]
        if not data_paths[0].exists():
            raise FileNotFoundError(f"You must have data for {group}=2.")
        if not data_paths[-1].exists():
            raise FileNotFoundError(f"You must have data for {group}={k}.")
        data_paths = [p for p in data_paths if p.exists()]
        data_paths = list(OrderedSet(data_paths))

    log.info("Constructing dataset from:")
    log.info("  " + "\n  ".join(map(str, data_paths)))

    if group.startswith("S5_only_swaps"):
        unique_tokens = (
            pl.read_csv(Path(data_dir) / "S5=2.csv")
            .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
            .explode("input")
            .unique()["input"]
            .to_list()
        )
    elif "limit_to" in group:
        g = group[:2]
        unique_tokens = (
            pl.read_csv(Path(data_dir) / f"{g}=2.csv")
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
        special_tokens=[(SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS))],
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

    if len(data_paths) == 1:
        dataset = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )
        if max_samples is not None:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
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


# =========================================================
# Classification metrics (existing)
# =========================================================

def compute_metrics(
    data: list[(Tensor, Tensor)],
    tokenizer: PreTrainedTokenizerFast,
    metric_fns: dict[str, Callable] = None,
    prefix: str | None = None,
) -> dict:
    if metric_fns is None:
        metric_fns = {
            "loss": ce_loss,
            "token_accuracy": token_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "sequence_accuracies": cumulative_sequence_accuracies,
        }

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


# =========================================================
# Regression (A5) fast target builder
# =========================================================

def a5_generators(device="cpu", dtype=torch.float64):
    """A=(0 1 2) and B=(0 1 2 3 4) in 3D icosahedral rep."""
    A = torch.tensor(
        [[0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    s5 = math.sqrt(5.0)
    B = torch.tensor(
        [
            [(s5 - 1) / 4, -(s5 + 1) / 4, 0.5],
            [(s5 + 1) / 4, 0.5, (s5 - 1) / 4],
            [-0.5, (s5 - 1) / 4, (s5 + 1) / 4],
        ],
        device=device,
        dtype=dtype,
    )
    return A, B


def build_group_representation_from_table(
    table: torch.Tensor,
    id_idx: int,
    gen_indices: dict[int, torch.Tensor],
    device="cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """BFS fill mats[g] = product of generators along Cayley edges."""
    N = int(table.shape[0])
    mats = torch.empty((N, 3, 3), device=device, dtype=dtype)
    known = torch.zeros((N,), device=device, dtype=torch.bool)

    mats[id_idx] = torch.eye(3, device=device, dtype=dtype)
    known[id_idx] = True

    q = [id_idx]
    gen_items = list(gen_indices.items())

    while q:
        x = q.pop(0)
        Mx = mats[x]
        for g_idx, Mg in gen_items:
            y = int(table[x, g_idx].item())
            if not bool(known[y].item()):
                mats[y] = Mx @ Mg
                known[y] = True
                q.append(y)

    if not bool(known.all().item()):
        missing = (~known).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Missing {len(missing)} elements in rep build, e.g. {missing[:10]}")
    return mats


def build_tokenid_to_mat_lookup_a5(a5_json: dict, tokenizer: PreTrainedTokenizerFast,
                                   device="cpu", dtype=torch.float64) -> torch.Tensor:
    """
    Returns L: [n_vocab_total,3,3] mapping token-id -> matrix.
    Identity for non-group tokens (BOS/PAD/etc).
    """
    elements = a5_json["elements"]
    if a5_json.get("table") is None:
        raise ValueError("A5 JSON table is missing. Provide full table (embedded or via --a5_json_path).")
    table = torch.tensor(a5_json["table"], dtype=torch.long, device=device)

    id_idx = elements.index("()")
    a_idx = elements.index("(0 1 2)")
    b_idx = elements.index("(0 1 2 3 4)")

    A, B = a5_generators(device=device, dtype=dtype)
    mats_by_elem = build_group_representation_from_table(
        table=table,
        id_idx=id_idx,
        gen_indices={a_idx: A, b_idx: B},
        device=device,
        dtype=dtype,
    )

    n_vocab_total = tokenizer.vocab_size + len(getattr(tokenizer, "added_tokens_encoder", {}))
    L = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(n_vocab_total, 1, 1)

    for elem_idx in range(len(elements)):
        tid = tokenizer.convert_tokens_to_ids(str(elem_idx))
        if tid is None or tid < 0 or tid >= n_vocab_total:
            continue
        L[tid] = mats_by_elem[elem_idx]

    return L


class GroupV0Dataset(Dataset):
    """
    Regression dataset wrapper:
      returns input_ids and a deterministic initial vector v0 (no target trajectory precomputation).
    Targets are built per-batch on GPU from the token->matrix lookup.
    """

    def __init__(
        self,
        base_dataset,
        seed: int = 0,
        sample_mode: str = "unit_sphere",   # "unit_sphere" or "gaussian"
        vec_dtype: torch.dtype = torch.float32,
    ):
        self.base = base_dataset
        self.seed = int(seed)
        self.sample_mode = sample_mode
        self.vec_dtype = vec_dtype

    def __len__(self):
        return len(self.base)

    def _sample_v0(self, idx: int) -> torch.Tensor:
        v = torch.tensor([1.0, 1.0, 1.0], dtype=self.vec_dtype)
        return v / (v.norm() + 1e-8)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        input_ids = item["input_ids"]
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        return {"input_ids": input_ids.to(torch.long), "v0": self._sample_v0(idx)}


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target: [B,T,3], mask: [B,T]"""
    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    diff2 = (pred - target).pow(2) * mask_f
    denom = (mask_f.sum().clamp_min(1.0) * pred.shape[-1])
    return diff2.sum() / denom


def _ensure_len_buffers(buf_sum: torch.Tensor, buf_cnt: torch.Tensor, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Grow SUM/COUNT buffers to length >= T."""
    if buf_sum is None:
        buf_sum = torch.zeros((T,), device=device, dtype=torch.float32)
        buf_cnt = torch.zeros((T,), device=device, dtype=torch.float32)
        return buf_sum, buf_cnt
    if buf_sum.numel() >= T:
        return buf_sum, buf_cnt
    new_sum = torch.zeros((T,), device=device, dtype=torch.float32)
    new_cnt = torch.zeros((T,), device=device, dtype=torch.float32)
    new_sum[: buf_sum.numel()] = buf_sum
    new_cnt[: buf_cnt.numel()] = buf_cnt
    return new_sum, new_cnt


def accumulate_per_position_mse(
    pred: torch.Tensor,   # [B,T,3]
    target: torch.Tensor, # [B,T,3]
    mask: torch.Tensor,   # [B,T]
    sum_buf: torch.Tensor,
    cnt_buf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate per-position MSE as SUM and COUNT (float32 tensors)."""
    with torch.no_grad():
        B, T, _ = pred.shape
        sum_buf, cnt_buf = _ensure_len_buffers(sum_buf, cnt_buf, T, pred.device)

        se = (pred.float() - target.float()).pow(2).mean(dim=-1)  # [B,T]
        m = mask.to(se.dtype)
        sum_buf[:T] += (se * m).sum(dim=0)
        cnt_buf[:T] += m.sum(dim=0)
    return sum_buf, cnt_buf


def finalize_per_position_curve(accelerator: Accelerator, sum_buf: torch.Tensor, cnt_buf: torch.Tensor) -> list[float]:
    """Reduce SUM/COUNT across processes and return per-position MSE list."""
    if sum_buf is None or cnt_buf is None or sum_buf.numel() == 0:
        return []

    sum_red = accelerator.reduce(sum_buf, reduction="sum")
    cnt_red = accelerator.reduce(cnt_buf, reduction="sum")

    denom = cnt_red.clamp_min(1.0)
    curve = (sum_red / denom).detach().cpu().tolist()
    return [float(x) for x in curve]


# =========================================================
# Output head swap for regression
# =========================================================

def _get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module):
    parent, leaf = _get_parent_module(root, module_name)
    setattr(parent, leaf, new_module)

def convert_model_to_regression_head(model: nn.Module, out_dim: int = 3) -> nn.Module:
    lm_head_candidates = []
    for name, mod in model.named_modules():
        if name.endswith("lm_head") and isinstance(mod, nn.Linear):
            lm_head_candidates.append((name, mod))

    if lm_head_candidates:
        # raise RuntimeError("No nn.Linear module named '*lm_head' found to replace.")
        name, head = lm_head_candidates[-1]
        new_head = nn.Linear(head.in_features, out_dim, bias=(head.bias is not None))
        _replace_module(model, name, new_head)
        return model

    # AUSSM-style: output
    output_candidates = []
    for name, mod in model.named_modules():
        # AUSSM shows: "model.output"
        if name.endswith("output") and isinstance(mod, nn.Linear):
            output_candidates.append((name, mod))
    if output_candidates:
        name, head = output_candidates[-1]
        new_head = nn.Linear(head.in_features, out_dim, bias=(head.bias is not None))
        _replace_module(model, name, new_head)
        return model

    raise RuntimeError(
        "Could not find a final projection to replace.\n"
        "Searched for modules ending with 'lm_head' or 'output'.\n"
    )

# =========================================================
# Helpers
# =========================================================

def make_job_name(hps: dict, priority=("model_name", "group", "n_layers"), exclude=()):
    parts = []
    for k in priority:
        if k in hps and k not in exclude:
            parts.append(f"{hps[k]}")
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
            f"min={t.min().item()} max={t.max().item()}"
        )


def max_prefix_at_threshold(seq_accs, threshold):
    seq_list = list(seq_accs) if seq_accs is not None else []
    max_len = 0
    for i, acc in enumerate(seq_list):
        if float(acc) >= threshold:
            max_len = i + 1
    return max_len


# =========================================================
# Train
# =========================================================

def train(
    # Task parameters
    task: str = "classification",          # "classification" or "a5_regression"
    a5_json_path: str | None = None,       # optional; if None uses embedded A5 json
    regression_sample_mode: str = "unit_sphere",  # "unit_sphere" or "gaussian"
    curriculum_loss_threshold_regression: float = 1e-3,

    # Data parameters
    group: str = "A5",
    k: int = 10,
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
    d_state: int = 16,
    mamba_expand_rate: int = 2,
    low_rank: int = 1,
    delta_rule_keys_activation: str = "id",
    activation_diag: str = "id",
    delta_rule_step_size_activation: str = "sigmoid",
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
    max_val_acc: float | None = 0.99,   # classification: accuracy stop; regression: loss stop (val/loss <= threshold)
    use_scheduler: bool = False,

    # Misc
    log_level: str = "INFO",
    seed: int = randint(0, 2**32 - 1),
    project_name: str = "word_problem_linrnnsimple",
    logging: bool = True,
    wandb_user: str = "irie",
    checkpoint_root: str = "run_checkpoints",
    save_every_steps: int = 1000,
    resume: bool = True,
    enable_mix_precision: bool = False,
):
    set_seed(seed)
    accelerator = Accelerator(log_with="wandb") if logging else Accelerator()
    log.setLevel(log_level)

    is_regression = (task.lower() == "a5_regression")

    # ----- Load dataset/tokenizer -----
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

    # Load test dataset
    if k_test is None or k_test == k:
        dataset_test = dataset
    else:
        datadict2 = get_dataset(
            group=group,
            k=k_test,
            strict_len=strict_len,
            train_size=train_size,
            data_dir=data_dir,
            supervised=tagging,
            max_samples=max_samples,
        )
        dataset_test = datadict2["dataset"]

    # ----- Regression setup (fast) -----
    tokenid_to_mat_gpu = None
    if is_regression:
        if a5_json_path is not None:
            with open(a5_json_path, "r") as f:
                a5_json = json.load(f)
        else:
            a5_json = A5_JSON_EMBEDDED

        tokenid_to_mat_cpu = build_tokenid_to_mat_lookup_a5(a5_json, tokenizer, device="cpu", dtype=torch.float64)

        if train_size < 1:
            train_base = dataset["train"]
            eval_base = dataset_test["test"]
        else:
            train_base = dataset
            eval_base = dataset_test

        train_ds = GroupV0Dataset(train_base, seed=seed, sample_mode=regression_sample_mode, vec_dtype=torch.float32)
        eval_ds = GroupV0Dataset(eval_base, seed=seed + 1, sample_mode=regression_sample_mode, vec_dtype=torch.float32)

        collate_fn = partial(pad_collate_v0, pad_token_id=tokenizer.pad_token_id)
    else:
        collate_fn = partial(pad_collate, pad_token_id=tokenizer.pad_token_id)

    # ----- Logger/hparams -----
    project_hps = {
        "task": task,
        "batch_size": batch_size,
        "betas": (beta1, beta2),
        "bias": bias,
        "compile": compile,
        "model_name": model_name,
        "hidden_size": hidden_size,
        "d_state": d_state,
        "mamba_expand_rate": mamba_expand_rate,
        "low_rank": low_rank,
        "n_train_samples": dataset["train"].num_rows if train_size < 1 else len(dataset),
        "n_test_samples": dataset_test["test"].num_rows if train_size < 1 else len(dataset_test),
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
        "regression_sample_mode": regression_sample_mode if is_regression else None,
        "a5_json_path": a5_json_path if is_regression else None,
    }

    exp_str = "".join(str(project_hps[k]) + "-" for k in project_hps.keys())
    exp_hash = str(int(hashlib.sha1(exp_str.encode("utf-8")).hexdigest(), 16) % (10**8))

    job_name = make_job_name(
        project_hps,
        priority=("task", "model_name", "group", "n_layers", "hidden_size", "lr", "seed", "k", "k_test", "mamba_expand_rate"),
        exclude={
            "tagging", "betas", "bias", "compile", "low_rank",
            "activation_diag", "curriculum", "strict_len", "eps",
            "normalize_recursion_output", "enable_mix_precision",
            "decoder_mode", "delta_rule_keys_activation",
            "delta_rule_step_size_activation",
            "use_scheduler", "max_samples",
            "train_size", "allow_neg_eigval",
            "n_test_samples", "n_train_samples",
            "max_val_acc", "a5_json_path"},
    )
    wandb_run_name = f"{job_name}-{exp_hash}"

    run_ckpt_dir = os.path.join(checkpoint_root, wandb_run_name, "latest")
    state_path = os.path.join(run_ckpt_dir, "trainer_state.json")
    has_ckpt = resume and os.path.isdir(run_ckpt_dir) and os.path.exists(state_path)

    accelerator.init_trackers(
        project_name,
        config=project_hps,
        init_kwargs={"wandb": {"entity": f"{wandb_user}", "name": wandb_run_name}},
    )

    log.info(f"Wandb run name: {wandb_run_name}")
    log.info(f"Config: {pformat(project_hps)}")
    log.info(f"Dataset (k={k}): {dataset}")
    log.info(f"Dataset test (k={k_test}): {dataset_test}")
    if is_regression:
        log.info("Task: A5 regression (fast GPU target build). Logging per-sequence-length MSE.")

    # ----- Construct model -----
    use_aussm = False
    if tagging:
        if model_name == "deltaproduct":
            print(f"Launching DeltaProduct-{num_householder} with {n_layers} layers and {n_heads} heads")
            conf = GatedDeltaProductConfig(
                vocab_size=n_vocab,
                use_short_conv=False,
                use_gate=False,
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

        elif model_name == "gla":
            print(f"Launching GLA with {n_layers} layers and {n_heads} heads")
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
            print(f"Launching Vanilla Transformer with {n_layers} layers and {n_heads} heads")
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
            print(f"Launching MambaLSTM with {n_layers} layers and {n_heads} heads")
            conf = MambaConfig(
                vocab_size=n_vocab,
                d_model=hidden_size,
                n_layer=n_layers,
                ssm_cfg={
                    "d_state": d_state,
                    "expand": mamba_expand_rate,
                    "d_conv": 4,
                    "positive_and_negative_associative_scan": False,
                },
            )
            model = MambaLSTM(conf)

        elif model_name == "neg_mamba_lstm":
            print(f"Launching PosNegative MambaLSTM with {n_layers} layers")
            conf = MambaConfig(
                vocab_size=n_vocab,
                d_model=hidden_size,
                n_layer=n_layers,
                ssm_cfg={
                    "d_state": d_state,
                    "expand": mamba_expand_rate,
                    "d_conv": 4,
                    "positive_and_negative_associative_scan": True,
                },
            )
            model = MambaLSTM(conf)

        elif model_name == "aussm":
            use_aussm = True
            print(f"Launching AUSSM with {n_layers} layers")
            assert n_layers > 0
            layer_str = "a"
            for _ in range(n_layers - 1):
                layer_str += "|a"
            conf = AussmConfig(
                d_model=hidden_size,
                vocab_size=n_vocab,
                output_vocab_size=n_vocab,
                layers=layer_str,
                d_state=d_state,
                mamba_expand=mamba_expand_rate,
                dt_rank="auto",
                d_conv=4,
                conv_bias=True,
                bias=False,
                ssmau_conv_1d=False,
                embedding_decay=False,
                ssmau_cuda=True,
                verbose=True,
            )
            model = AUSSM(conf)

        else:
            raise ValueError(f"model_name: {model_name} not supported.")

        if is_regression:
            # if use_aussm:
            #     raise RuntimeError("A5 regression not supported for AUSSM in this script.")
            model = convert_model_to_regression_head(model, out_dim=3)

        if not use_aussm:
            model.to(torch.bfloat16)
        model.to("cuda")

    if compile:
        torch.set_float32_matmul_precision("high")
        log.info("Compiling model...")
        model = torch.compile(model, dynamic=True)
        log.info("Model compiled!")

    log.info(f"Model: {model}")
    log.info(f"Number of parameters: {humanize.intword(model.num_parameters)} ({model.num_parameters})")
    log.info(f"Accelerator state: {accelerator.state}")

    device = accelerator.device
    model = model.to(device)

    if is_regression:
        # Move lookup to GPU once
        tokenid_to_mat_gpu = tokenid_to_mat_cpu.float().to(device)  # [V,3,3]

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=op_eps,
        weight_decay=weight_decay,
    )

    # ----- Dataloaders -----
    if is_regression:
        train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
        eval_dataloader = DataLoader(eval_ds, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    else:
        if train_size < 1:
            train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
            eval_dataloader = DataLoader(dataset_test["test"], shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
        else:
            train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
            eval_dataloader = DataLoader(dataset_test, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_dataloader))
    model, optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, eval_dataloader
    )

    # Classification metric fns
    metric_fns = {
        "loss": ce_loss,
        "sequence_accuracy": token_accuracy,
    }
    if tagging and not is_regression:
        metric_fns["sequence_accuracy"] = sequence_accuracy
        metric_fns["token_accuracy"] = token_accuracy
        metric_fns["sequence_accuracies"] = cumulative_sequence_accuracies

    # ----- Resume state -----
    global_step = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")  # regression
    start_epoch = 0

    THRESHOLD_90 = 0.90
    THRESHOLD_99 = 0.99
    best_train_maxlen_at_90 = 0
    best_val_maxlen_at_90 = 0
    best_train_maxlen_at_99 = 0
    best_val_maxlen_at_99 = 0

    train_dataloader_base = train_dataloader
    curriculum_idx = (min(2, k + 1) if curriculum else k + 1)
    resume_step_in_epoch = 0

    if has_ckpt:
        accelerator.print(f"Resuming from checkpoint: {run_ckpt_dir}")
        accelerator.load_state(run_ckpt_dir)

        with open(state_path, "r") as f:
            st = json.load(f)

        global_step = int(st.get("global_step", 0))
        curriculum_idx = int(st.get("curriculum_idx", curriculum_idx))
        start_epoch = int(st.get("epoch", global_step // len(train_dataloader)))

        if is_regression:
            best_val_loss = float(st.get("best_val_loss", best_val_loss))
        else:
            best_val_acc = float(st.get("best_val_acc", best_val_acc))

        steps_per_epoch = len(train_dataloader)
        resume_step_in_epoch = global_step % steps_per_epoch

    print(f"Starting training (task={task}) (start_curriculum_length={curriculum_idx}, k={k})")

    try:
        for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", position=0, leave=False):
            model.train()

            if epoch == start_epoch and resume_step_in_epoch > 0:
                epoch_dataloader = accelerator.skip_first_batches(train_dataloader_base, resume_step_in_epoch)
                accelerator.print(f"Epoch {epoch}: skipping first {resume_step_in_epoch} batches")
            else:
                epoch_dataloader = train_dataloader_base

            train_results = []
            epoch_train_loss = 0.0
            n_batches = 0

            # Regression per-position accumulators (fast)
            train_pos_sum = None
            train_pos_cnt = None

            for batch in (t_bar := tqdm(epoch_dataloader, desc="Train", position=1, leave=False)):
                global_step += 1
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_mix_precision):
                    if is_regression:
                        ids = batch["input_ids"].to(device)   # [B,T]
                        mask = batch["mask"].to(device)       # [B,T]
                        v = batch["v0"].to(device)            # [B,3]

                        # curriculum crop on time dimension
                        if curriculum_idx is not None:
                            ids_used = ids[:, :curriculum_idx]
                            mask_used = mask[:, :curriculum_idx]
                        else:
                            ids_used = ids
                            mask_used = mask

                        # Ms: [B,T,3,3]
                        Ms = tokenid_to_mat_gpu[ids_used]

                        # Build targets trajectory on GPU (O(T) loop)
                        targets_list = []
                        vv = v
                        T_used = Ms.shape[1]
                        for t in range(T_used):
                            vv = torch.bmm(Ms[:, t], vv.unsqueeze(-1)).squeeze(-1)  # [B,3]
                            targets_list.append(vv)
                        targets = torch.stack(targets_list, dim=1)  # [B,T,3]

                        if use_aussm:
                            slen = ids_used.shape[-1]
                            if slen % 32 != 0:
                                assert (slen - 1) % 32 == 0, f"slen = {slen}"
                                ids_used = ids_used[:, :slen - 1]
                                mask_used = mask_used[:, :slen - 1]
                                targets = targets[:, :slen - 1]
                                warnings.warn(f"[AUSSM] Seq len is {slen}, cut it to {slen - 1}", RuntimeWarning)

                        # Model output [B,T,3]
                        out = model(ids_used)

                        loss = masked_mse_loss(out.float(), targets.float(), mask_used)

                        # accumulate per-position MSE curves cheaply
                        train_pos_sum, train_pos_cnt = accumulate_per_position_mse(
                            out, targets, mask_used, train_pos_sum, train_pos_cnt
                        )

                    else:
                        source = batch["input_ids"]
                        target = batch["labels"]

                        if use_aussm:
                            slen = source.shape[-1]
                            if slen % 32 != 0:
                                assert (slen - 1) % 32 == 0, f"slen = {slen}"
                                source = source[:, :slen - 1]
                                target = target[:, :slen - 1]
                                warnings.warn(f"[AUSSM] Seq len is {slen}, cut it to {slen - 1}", RuntimeWarning)

                        output = model(source)
                        out_used = output[:, :curriculum_idx, :]
                        tgt_used = target[:, :curriculum_idx]

                        preds_used, refs_used = accelerator.gather_for_metrics((out_used, tgt_used))
                        metrics = compute_metrics(
                            [(preds_used, refs_used)],
                            tokenizer=tokenizer,
                            metric_fns=metric_fns,
                            prefix="train",
                        )
                        train_results.append(metrics)

                        loss = F.cross_entropy(out_used.flatten(end_dim=-2), tgt_used.flatten())

                check_nan("loss", loss)

                epoch_train_loss += float(loss.item())
                n_batches += 1

                accelerator.backward(loss)
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, norm_type=2.0)
                optimizer.step()

                if use_scheduler:
                    scheduler.step()

                should_save = (global_step % save_every_steps == 0)

                if accelerator.is_main_process and should_save:
                    os.makedirs(os.path.dirname(run_ckpt_dir), exist_ok=True)

                    tmp_dir = run_ckpt_dir + ".tmp"
                    bak_dir = run_ckpt_dir + ".bak"

                    if os.path.isdir(tmp_dir):
                        shutil.rmtree(tmp_dir)

                    accelerator.save_state(tmp_dir)

                    tmp_state_path = os.path.join(tmp_dir, "trainer_state.json")
                    with open(tmp_state_path, "w") as f:
                        payload = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "curriculum_idx": curriculum_idx,
                        }
                        if is_regression:
                            payload["best_val_loss"] = best_val_loss
                        else:
                            payload["best_val_acc"] = best_val_acc
                        json.dump(payload, f, indent=2)

                    try:
                        if os.path.isdir(bak_dir):
                            shutil.rmtree(bak_dir)
                        if os.path.isdir(run_ckpt_dir):
                            os.rename(run_ckpt_dir, bak_dir)
                        os.rename(tmp_dir, run_ckpt_dir)
                        if os.path.isdir(bak_dir):
                            shutil.rmtree(bak_dir)
                    except Exception:
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

            # ----- Log train metrics -----
            if is_regression:
                train_curve = finalize_per_position_curve(accelerator, train_pos_sum, train_pos_cnt)
                train_loss_scalar = float(epoch_train_loss)

                train_metrics = {
                    "train/loss": train_loss_scalar,
                    "train/sequence_errors": train_curve,
                    "train/sequence_error": float(sum(train_curve) / max(len(train_curve), 1)) if train_curve else 0.0,
                }
                accelerator.log(train_metrics, step=global_step)

                # Plot train per-length MSE
                if train_curve and accelerator.is_main_process:
                    train_table = wandb.Table(
                        data=[[i + 1, float(e)] for i, e in enumerate(train_curve)],
                        columns=["sequence_length", "mse"],
                    )
                    accelerator.log(
                        {"train_sequence_error_table": wandb.plot.line(
                            train_table, "sequence_length", "mse", title="Train per-length MSE"
                        )},
                        step=global_step,
                    )
            else:
                accelerator.log(reduce_metrics(train_results), step=global_step)

            # ----- Curriculum update -----
            if curriculum and curriculum_idx < k + 1:
                if (not is_regression and epoch_train_loss < 0.3) or (is_regression and epoch_train_loss < curriculum_loss_threshold_regression):
                    curriculum_idx += 1
                    print(f" Increasing curriculum index to {curriculum_idx}")

            # ----- Eval -----
            model.eval()
            eval_results = []
            eval_pos_sum = None
            eval_pos_cnt = None
            eval_loss_sum = 0.0
            eval_loss_batches = 0

            for batch in tqdm(eval_dataloader, desc="Eval", position=1, leave=False):
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_mix_precision):
                        if is_regression:
                            ids = batch["input_ids"].to(device)
                            mask = batch["mask"].to(device)
                            v = batch["v0"].to(device)

                            Ms = tokenid_to_mat_gpu[ids]
                            targets_list = []
                            vv = v
                            T_used = Ms.shape[1]
                            for t in range(T_used):
                                vv = torch.bmm(Ms[:, t], vv.unsqueeze(-1)).squeeze(-1)
                                targets_list.append(vv)
                            targets = torch.stack(targets_list, dim=1)

                            if use_aussm:
                                slen = ids.shape[-1]
                                if slen % 32 != 0:
                                    assert (slen - 1) % 32 == 0, f"slen = {slen}"
                                    ids = ids[:, :slen - 1]
                                    mask = mask[:, :slen - 1]
                                    targets = targets[:, :slen - 1]
                                    warnings.warn(f"[AUSSM] Seq len is {slen}, cut it to {slen - 1}", RuntimeWarning)

                            out = model(ids)
                            loss = masked_mse_loss(out.float(), targets.float(), mask)

                            eval_pos_sum, eval_pos_cnt = accumulate_per_position_mse(
                                out, targets, mask, eval_pos_sum, eval_pos_cnt
                            )
                            eval_loss_sum += float(loss.item())
                            eval_loss_batches += 1
                        else:
                            source = batch["input_ids"]
                            target = batch["labels"]

                            if use_aussm:
                                slen = source.shape[-1]
                                if slen % 32 != 0:
                                    assert (slen - 1) % 32 == 0, f"slen = {slen}"
                                    source = source[:, :slen - 1]
                                    target = target[:, :slen - 1]
                                    warnings.warn(f"[AUSSM] Seq len is {slen}, cut it to {slen - 1} ", RuntimeWarning)

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

            # ----- Reduce + log eval -----
            if is_regression:
                val_curve = finalize_per_position_curve(accelerator, eval_pos_sum, eval_pos_cnt)
                val_loss = float(eval_loss_sum / max(eval_loss_batches, 1))

                eval_metrics = {
                    "val/loss": val_loss,
                    "val/sequence_errors": val_curve,
                    "val/sequence_error": float(sum(val_curve) / max(len(val_curve), 1)) if val_curve else 0.0,
                }
                accelerator.log(eval_metrics, step=global_step)

                # Plot val per-length MSE
                if val_curve and accelerator.is_main_process:
                    val_table = wandb.Table(
                        data=[[i + 1, float(e)] for i, e in enumerate(val_curve)],
                        columns=["sequence_length", "mse"],
                    )
                    accelerator.log(
                        {"val_sequence_error_table": wandb.plot.line(
                            val_table, "sequence_length", "mse", title="Val per-length MSE"
                        )},
                        step=global_step,
                    )

                # Save best by val loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_pretrained(
                        accelerator=accelerator,
                        model=model,
                        group=group,
                        model_name=f"{model_name}_a5reg",
                        k=k,
                        num_householders=num_householder,
                        save_root="checkpoints",
                        seed=seed,
                    )
                accelerator.log({"val/best_loss": best_val_loss}, step=global_step)

            else:
                eval_metrics = reduce_metrics(eval_results)
                accelerator.log(eval_metrics, step=global_step)

                train_metrics = reduce_metrics(train_results)

                train_maxlen_at_90 = max_prefix_at_threshold(train_metrics.get("train/sequence_accuracies", []), THRESHOLD_90)
                val_maxlen_at_90 = max_prefix_at_threshold(eval_metrics.get("val/sequence_accuracies", []), THRESHOLD_90)
                train_maxlen_at_99 = max_prefix_at_threshold(train_metrics.get("train/sequence_accuracies", []), THRESHOLD_99)
                val_maxlen_at_99 = max_prefix_at_threshold(eval_metrics.get("val/sequence_accuracies", []), THRESHOLD_99)

                accelerator.log(
                    {
                        "train/max_seq_len_at_90": int(train_maxlen_at_90),
                        "val/max_seq_len_at_90": int(val_maxlen_at_90),
                        "train/max_seq_len_at_99": int(train_maxlen_at_99),
                        "val/max_seq_len_at_99": int(val_maxlen_at_99),
                    },
                    step=global_step,
                )

                best_train_maxlen_at_90 = max(best_train_maxlen_at_90, int(train_maxlen_at_90))
                best_val_maxlen_at_90 = max(best_val_maxlen_at_90, int(val_maxlen_at_90))
                best_train_maxlen_at_99 = max(best_train_maxlen_at_99, int(train_maxlen_at_99))
                best_val_maxlen_at_99 = max(best_val_maxlen_at_99, int(val_maxlen_at_99))

                accelerator.log(
                    {
                        "best/train_max_seq_len_at_90": int(best_train_maxlen_at_90),
                        "best/val_max_seq_len_at_90": int(best_val_maxlen_at_90),
                        "best/train_max_seq_len_at_99": int(best_train_maxlen_at_99),
                        "best/val_max_seq_len_at_99": int(best_val_maxlen_at_99),
                    },
                    step=global_step,
                )

                # Train accuracy table
                if "train/sequence_accuracies" in train_metrics and accelerator.is_main_process:
                    train_seq_vals = train_metrics["train/sequence_accuracies"]
                    train_table = wandb.Table(
                        data=[[i + 1, float(acc)] for i, acc in enumerate(train_seq_vals)],
                        columns=["sequence_length", "accuracy"],
                    )
                    accelerator.log(
                        {"train_sequence_accuracy_table": wandb.plot.line(
                            train_table, "sequence_length", "accuracy", title="Train sequence accuracy"
                        )},
                        step=global_step,
                    )

                # Save best by val accuracy
                if eval_metrics["val/sequence_accuracy"] > best_val_acc:
                    best_val_acc = eval_metrics["val/sequence_accuracy"]
                    save_pretrained(
                        accelerator=accelerator,
                        model=model,
                        group=group,
                        model_name=model_name,
                        k=k,
                        num_householders=num_householder,
                        save_root="checkpoints",
                        seed=seed,
                    )
                eval_metrics["val/best_sequence_accuracy"] = best_val_acc
                accelerator.log(eval_metrics, step=global_step)

                if max_val_acc is not None and best_val_acc >= max_val_acc:
                    log.info(f"Validation accuracy reached {max_val_acc}. Stopping training.")
                    break

        accelerator.end_training()

    except Exception as e:
        wandb.alert(title="Training Error", text=f"An error occurred: {str(e)}")
        import traceback
        wandb.log({"error_traceback": traceback.format_exc()})
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire()

