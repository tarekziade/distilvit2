import os
import sys
import shutil
import platform

# Multi-GPU configuration for Windows
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    # Windows requires Gloo backend (NCCL not available)
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"

environ_dict = {"NCCL_P2P_DISABLE": "1",
                "NCCL_IB_DISABLE": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "WANDB_PROJECT": "distilvit",
                "WANDB_LOG_MODEL": "false"
                }

from functools import partial
import torch
from collections.abc import Mapping
import argparse

import nltk
import evaluate
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoTokenizer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import concatenate_datasets, DatasetDict
from transformers.trainer_callback import EarlyStoppingCallback

from distilvit._datasets import DATASETS
from distilvit.prefix_model import PrefixConditioningVLM, PrefixConditioningConfig
# Temporarily disabled due to torch/optimum compatibility - quantize after training manually
# from distilvit.quantize import main as quantize
from distilvit.upload import push_to_hub


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log_gpu_info():
    """Log GPU information for multi-GPU setups"""
    if not torch.cuda.is_available():
        print("CUDA not available - running on CPU or MPS")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"GPU Configuration:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Current device: {torch.cuda.current_device()}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")

    if IS_WINDOWS and num_gpus > 1:
        print(f"\n  Multi-GPU on Windows: Using Gloo backend (NCCL not available)")
        print(f"  Expected speedup: ~1.6-1.8x with {num_gpus} GPUs")

    print(f"{'='*60}\n")


def get_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)


# Default max length for captions - optimized for short, concise captions (25-30 tokens)
DEFAULT_MAX_LENGTH = 30
THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING = 42
MODEL_ID = "mozilla/distilvit"

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            with open(self.file_path, "a") as f:
                f.write(f"{metrics}\n")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_metrics(
    tokenizer,
    rouge,
    meteor,
    eval_preds,
    args=None,
):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    if args.debug:
        for expected, res in zip(decoded_labels, decoded_preds):
            print(f"Expected: {expected}, got: {res}")

    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["meteor"] = meteor.compute(
        predictions=decoded_preds, references=decoded_labels
    )["meteor"]

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return result


def freeze_model_layers(model, freeze_encoder_layers=3, freeze_decoder_layers=3):
    for i, layer in enumerate(model.encoder.encoder.layer):
        if i < freeze_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

    for i, layer in enumerate(model.decoder.transformer.h):
        if i < freeze_decoder_layers:
            for param in layer.parameters():
                param.requires_grad = False


def data_collator(tokenizer, max_length, features):
    """
    Simple, robust data collator for prefix-conditioning model.
    Extracts only pixel_values and labels, ignoring everything else.
    """
    import torch

    # Convert to list of dicts if needed
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]

    # Extract pixel_values - must be present
    pixel_values = []
    for f in features:
        pv = f.get("pixel_values")
        if pv is not None:
            # Convert to tensor if needed
            if isinstance(pv, np.ndarray):
                pv = torch.from_numpy(pv)
            elif isinstance(pv, list):
                pv = torch.tensor(pv, dtype=torch.float32)
            elif not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            pixel_values.append(pv)

    if not pixel_values:
        raise ValueError("No pixel_values found in batch!")

    # Extract and process labels
    labels = []
    input_ids = []

    for f in features:
        label = f.get("labels", [])
        if isinstance(label, (torch.Tensor, np.ndarray)):
            label = label.tolist() if hasattr(label, 'tolist') else list(label)

        # Pad or truncate
        if len(label) > max_length:
            label = label[:max_length]
        elif len(label) < max_length:
            label = label + [tokenizer.pad_token_id] * (max_length - len(label))

        labels.append(label)
        # For causal LM training, input_ids are the same as labels
        # The model will handle the shifting internally
        input_ids.append(label)

    # Stack into batch tensors
    batch = {
        "pixel_values": torch.stack(pixel_values),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    return batch


def get_arg_parser(root_dir=None):
    if root_dir is None:
        root_dir = os.path.join(os.path.dirname(__file__), "..")

    parser = argparse.ArgumentParser(
        description="Train a Vision Encoder Decoder Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        type=str,
        help="Model ID",
    )

    parser.add_argument(
        "--sample",
        default=None,
        type=int,
        help="Sample data",
    )

    parser.add_argument(
        "--tag",
        type=str,
        help="HF tag",
        default=None,
    )

    parser.add_argument(
        "--save-dir",
        default=root_dir,
        type=str,
        help="Save dir",
    )

    parser.add_argument(
        "--cache-dir",
        default=os.path.join(root_dir, "cache"),
        type=str,
        help="Cache dir",
    )

    parser.add_argument(
        "--prune-cache",
        default=False,
        action="store_true",
        help="Empty cache dir",
    )

    parser.add_argument(
        "--checkpoints-dir",
        default=os.path.join(root_dir, "checkpoints"),
        type=str,
        help="Checkpoints dir",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode",
    )

    parser.add_argument(
        "--num-train-epochs", type=int, default=3, help="Number of epochs"
    )

    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Save steps")

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum length for generated captions",
    )

    parser.add_argument(
        "--encoder-model",
        default="google/siglip-base-patch16-224",
        type=str,
        help="Base model for the encoder (default: SigLIP-2 Base)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        type=str,
        help="Base model to train again from",
    )
    parser.add_argument(
        "--device",
        default=get_device(),
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Base model to train again from",
    )

    parser.add_argument(
        "--base-model-revision",
        default=None,
        type=str,
        help="Base model revision",
    )

    parser.add_argument("--push-to-hub", action="store_true", help="Push to hub")

    parser.add_argument(
        "--feature-extractor-model",
        default="google/siglip-base-patch16-224",
        type=str,
        help="Feature extractor model for the encoder (default: SigLIP-2 Base)",
    )
    parser.add_argument(
        "--decoder-model",
        default="HuggingFaceTB/SmolLM-135M",
        type=str,
        help="Model for the decoder (default: SmolLM-135M). With prefix-conditioning + LoRA, any decoder-only LM works (SmolLM, GPT2, OPT, Llama-based models).",
    )

    # LoRA and freezing configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16, range: 8-32)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling parameter (default: 16)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=True,
        help="Freeze vision encoder (default: True)",
    )
    parser.add_argument(
        "--unfreeze-vision-layers",
        type=int,
        default=0,
        help="Number of last vision encoder layers to unfreeze (default: 0)",
    )
    parser.add_argument(
        "--projection-type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Type of projection head: linear or mlp (default: linear)",
    )
    parser.add_argument(
        "--projection-lr",
        type=float,
        default=1e-3,
        help="Learning rate for projection head (default: 1e-3)",
    )
    parser.add_argument(
        "--lora-lr",
        type=float,
        default=5e-5,
        help="Learning rate for LoRA parameters (default: 5e-5)",
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Dataset to use for training",
    )
    return parser

def parse_args(arg_list=None):
    parser = get_arg_parser()
    return parser.parse_args(arg_list)


def train(args):
    # Log GPU configuration first
    log_gpu_info()

    get_nltk()
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # Load image processor
    feature_extractor = AutoImageProcessor.from_pretrained(args.feature_extractor_model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)

    # Configure tokenizer for decoder-only models
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if tokenizer.bos_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.bos_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'bos_token': '[BOS]'})

    # Check if loading from existing model
    if args.base_model:
        print(f"Loading existing model from {args.base_model}")
        model = PrefixConditioningVLM.from_pretrained(args.base_model)
        model_name = f"{args.base_model}+fine-tuned"
    else:
        # Load vision encoder
        print(f"Loading vision encoder: {args.encoder_model}")
        if 'siglip' in args.encoder_model.lower():
            from transformers import SiglipVisionModel
            vision_encoder = SiglipVisionModel.from_pretrained(args.encoder_model)
        else:
            from transformers import AutoModel
            vision_encoder = AutoModel.from_pretrained(args.encoder_model)

        # Load language model
        print(f"Loading language model: {args.decoder_model}")
        language_model = AutoModelForCausalLM.from_pretrained(args.decoder_model)

        # Resize embeddings if we added special tokens
        language_model.resize_token_embeddings(len(tokenizer))

        # Create prefix-conditioning config
        config = PrefixConditioningConfig(
            vision_config=vision_encoder.config,
            text_config=language_model.config,
            projection_dim=512,
            projection_type=args.projection_type,
            mlp_hidden_dim=2048,
            freeze_vision_encoder=args.freeze_vision,
            vision_unfreeze_layers=args.unfreeze_vision_layers,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_length=args.max_length,
        )

        # Create model with prefix conditioning + LoRA
        print("Creating prefix-conditioning model with LoRA adapters...")
        model = PrefixConditioningVLM(config, vision_encoder, language_model)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model.language_model, 'gradient_checkpointing_enable'):
            model.language_model.gradient_checkpointing_enable()

        # Store tokenizer in model for generation
        model.tokenizer = tokenizer

        # Print trainable parameters
        model.print_trainable_parameters()

        model_name = (
            f"{args.encoder_model.split('/')[-1]}-{args.decoder_model.split('/')[-1]}-lora"
        )

    # Move to device
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")
    model.to(args.device)

    save_path = os.path.join(args.save_dir, model_name)

    print("Sources", args.dataset)
    datasets = []
    for name in args.dataset:
        get_dataset = DATASETS[name]
        datasets.append(
            get_dataset(
                args.feature_extractor_model,
                args.decoder_model,
                args=args,
            )
        )

    print("Datasets loaded", datasets)
    combined = DatasetDict()
    for split in datasets[0].keys():
        combined[split] = concatenate_datasets([ds[split] for ds in datasets])

    ds = combined.shuffle(seed=THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING)

    print("Datasets combined and shuffled", ds)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # Detect multi-GPU setup
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    is_multi_gpu = num_gpus > 1

    # Optimize batch sizes for available hardware
    if is_multi_gpu:
        # Multi-GPU (e.g., 2xRTX4090): Use larger per-device batch with less accumulation
        per_device_batch = 16  # 16 per GPU = 32 total per step
        gradient_accumulation = 2  # Effective batch = 32 * 2 = 64
        num_workers = 4  # Per GPU
    elif torch.cuda.is_available():
        # Single GPU: Smaller batch with more accumulation
        per_device_batch = 8  # RTX 4090 can handle this
        gradient_accumulation = 8  # Effective batch = 64
        num_workers = 4
    else:
        # MPS or CPU: Conservative settings
        per_device_batch = 4
        gradient_accumulation = 16  # Effective batch = 64
        num_workers = 2 if IS_WINDOWS else 4

    # Training arguments optimized for LoRA + prefix conditioning
    training_args = TrainingArguments(
        output_dir=args.checkpoints_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=args.lora_lr,  # Base LR for LoRA parameters
        weight_decay=0.01,  # As specified
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_steps=500,  # As specified (500-1000 range)
        lr_scheduler_type="cosine",  # Cosine decay as specified
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(args.checkpoints_dir, "logs"),
        logging_steps=10,
        report_to="none",  # Disable wandb for now
        fp16=torch.cuda.is_available(),  # Enable fp16 on CUDA (RTX 4090 supports it)
        bf16=False,
        gradient_checkpointing=False,  # Disabled for custom model (enabled directly on language_model)
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory for CUDA
        remove_unused_columns=False,  # Important: keep pixel_values
        # Multi-GPU specific settings
        ddp_backend="gloo" if (IS_WINDOWS and is_multi_gpu) else None,
        ddp_find_unused_parameters=False,  # Faster training, safe for our model
        local_rank=-1,  # Auto-detect in distributed setup
    )

    print(f"\nTraining configuration:")
    print(f"  Per-device batch size: {per_device_batch}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Effective batch size: {per_device_batch * gradient_accumulation * num_gpus}")
    print(f"  Dataloader workers: {num_workers}")
    print(f"  FP16: {training_args.fp16}")
    if is_multi_gpu:
        print(f"  Multi-GPU: {num_gpus} GPUs")
        print(f"  DDP backend: {training_args.ddp_backend or 'nccl'}")
    print()

    # Create optimizer with different learning rates for projection vs LoRA
    # Projection head gets higher LR, LoRA gets lower LR
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "projection" in n and p.requires_grad],
            "lr": args.projection_lr,
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad],
            "lr": args.lora_lr,
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if "vision" in n and p.requires_grad],
            "lr": args.lora_lr * 0.1,  # Even lower LR for unfrozen vision layers
            "weight_decay": 0.01,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups)

    last_checkpoint = get_last_checkpoint(args.checkpoints_dir)
    metrics_logger_callback = MetricsLoggerCallback(
        os.path.join(args.checkpoints_dir, "metrics.txt")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=partial(data_collator, tokenizer, args.max_length),
        optimizers=(optimizer, None),  # Custom optimizer, default scheduler
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            metrics_logger_callback,
        ],
    )

    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # Quantization temporarily disabled due to torch/optimum compatibility
    # Run quantization manually after training: bin/python distilvit/quantize.py --model_id MODEL_PATH --quantize --task image-to-text-with-past
    # q_args = [
    #     "quantize",
    #     "--model_id",
    #     save_path,
    #     "--quantize",
    #     "--task",
    #     "image-to-text-with-past",
    # ]
    # old = sys.argv
    # sys.argv = q_args
    # try:
    #     quantize()
    # finally:
    #     sys.argv = old

    print(f"Model saved to {save_path}. You may need to copy in model card in docs directory.")

    if args.push_to_hub:
        push_to_hub(args.model_id, save_path, args.tag, "New training")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
